import logging
import queue
import sys
import threading
import traceback

import numpy as np
from gameanalysis import collect
from gameanalysis import nash
from gameanalysis import reduction
from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils


_log = logging.getLogger(__name__)


# TODO Split outer loop from inner loop so that arbitrary best response / next
# strategy oracles can be used. This change would require a significant
# restructure in order to preserve existing data.


def outer_loop(prof_sched, game, initial_subgame=None, *, red=None,
               regret_thresh=1e-3, dist_thresh=1e-3, max_resamples=10,
               subgame_size=3, num_equilibria=1, num_backups=1,
               devs_by_role=False):
    """Outerloop a game using a scheduler

    If initial_subgame is unspecified, this will instead perform only the inner
    loop of the full game specified.

    Parameters
    ----------
    prof_sched : Scheduler
        The scheduler used to generate payoff data for profiles.
    game : BaseGame
        The gameanalysis basegame for the game to find equilibria on.
    initial_subgame : subgame array, optional
        The initial subgame to innerloop before checking for deviations
        outside. If unspecified, the whole game will be used, effectively
        making this one iteration of the inner loop with no outer looping.
    red : Reduction, optional
        The reduction to apply to the game before scheduling. Using a reduction
        will sample fewer profiles, while only approximating equilibria.
    regret_thresh : float, optional
        The maximum regret to consider an equilibrium an equilibrium.
    dist_thresh : float, optional
        The minimum norm between two mixed profiles for them to be considered
        distinct.
    max_resamples : int > 0, optional
        The maximum number of times to resample a subgame when no equilibria
        can be found before giving up.
    subgame_size : int > 0, optional
        The maximum subgame support size with which beneficial deviations must
        be explored. Subgames with support larger than this are queued and only
        explored in the event that no equilibrium can be found in beneficial
        deviations smaller than this.
    num_equilibria : int > 0, optional
        The number of equilibria to attempt to find. Only one is guaranteed,
        but this might be beneifical if the game has a known degenerate
        equilibria, but one which is still helpful as a deviating strategy.
    num_backups : int > 0, optional
        In the event that no equilibrium can be found in beneficial deviations
        to small subgames, other subgames will be explored. This parameter
        indicates how many subgames for each role should be explored.
    devs_by_role : boolean, optional
        If specified, deviations will only be explored for each role in order,
        proceeding to the next role only when no beneficial deviations are
        found. This can reduce the number of profiles sampled, but may also
        fail to find certain equilibria due to the different path through
        subgames.
    """
    if red is None:
        red = reduction.Identity(game.num_strategies, game.num_players)
    assert rsgame.basegame_copy(game) == red.full_game, \
        "reduction must match game format"

    return _OuterLoop(prof_sched, red, regret_thresh=regret_thresh,
                      dist_thresh=dist_thresh, max_resamples=max_resamples,
                      subgame_size=subgame_size, num_equilibria=num_equilibria,
                      num_backups=num_backups,
                      devs_by_role=devs_by_role).run(initial_subgame)


class _OuterLoop(object):
    """Object to keep track of outer loop progress"""

    def __init__(self, prof_sched, reduction, regret_thresh, dist_thresh,
                 max_resamples, subgame_size, num_equilibria, num_backups,
                 devs_by_role):
        # Input data
        self._sched = _Scheduler(prof_sched, reduction)
        self._game = reduction.red_game
        self._full_sched = self._sched.get_subgame_scheduler(
            np.ones(self._game.num_role_strats, bool))
        self._init_role_dev = 0 if devs_by_role else None
        self._num_eqa = num_equilibria
        self._regret_thresh = regret_thresh
        self._params = dict(
            regret_thresh=regret_thresh, dist_thresh=dist_thresh,
            max_resamples=max_resamples, subgame_size=subgame_size,
            num_equilibria=num_equilibria, num_backups=num_backups,
            devs_by_role=devs_by_role)

        # Bookkeeping
        self._equilibria = []
        self._best_responses = []
        self._threads = queue.Queue()
        self._run_lock = threading.Lock()
        self._thread_failed = False

    def _add_deviations(self, mix, role_index):
        _log.info('scheduling outerloop deviations from %s for role %s',
                  mix, role_index)
        thread = threading.Thread(
            target=lambda: self._run_deviations(mix, role_index))
        thread.start()
        self._threads.put(thread)

    def _run_deviations(self, mix, role_index):
        try:
            support = mix > 0
            game = self._full_sched.get_deviations(support, 1, role_index)
            gains = regret.mixture_deviation_gains(game, mix)
            if role_index is None:
                assert not np.isnan(gains).any(), "There were nan regrets"
                if np.all(gains <= self._regret_thresh):  # Found equilibrium
                    self._equilibria.append(mix)  # atomic
                else:
                    for rgains, rs in zip(game.role_split(gains),
                                          game.role_starts):
                        br = np.argmax(rgains)
                        self._best_responses.append(rs + br)  # atomic

            else:  # Set role index
                rgains = game.role_split(gains)[role_index]
                assert not np.isnan(rgains).any(), "There were nan regrets"
                if np.all(rgains <= self._regret_thresh):  # No deviations
                    role_index += 1
                    if role_index < game.num_roles:  # Explore next deviation
                        self._add_deviations(mix, role_index)
                    else:  # found equilibrium
                        self._equilibria.append(mix)  # atomic
                else:
                    br = game.role_starts[role_index] + np.argmax(rgains)
                    self._best_responses.append(br)  # atomic
        except Exception as ex:  # pragma: no cover
            self._thread_failed = True
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            raise ex

    def run(self, initial_subgame=None):
        last_mask = np.zeros(self._game.num_role_strats, bool)
        mask = (np.ones(self._game.num_role_strats, bool)
                if initial_subgame is None else initial_subgame)

        with self._run_lock:
            assert self._threads.empty()
            self._equilibria.clear()
            self._thread_failed = False

            while (len(self._equilibria) < self._num_eqa
                   and not last_mask.all()):
                # Generate view
                np.copyto(last_mask, mask)
                self._equilibria.clear()
                self._best_responses.clear()
                sub_sched = self._sched.get_subgame_scheduler(mask)
                sub_game = subgame.subgame(self._game, mask)

                # get inner loop equilibria and
                eqa = _InnerLoop(sub_sched, sub_game, **self._params).run()
                for eqm in eqa:
                    self._add_deviations(subgame.translate(eqm, mask),
                                         self._init_role_dev)
                while not self._threads.empty():
                    if self._thread_failed:
                        raise RuntimeError("a thread failed, check log for details")  # pragma: no cover # noqa
                    else:
                        self._threads.get().join()

                # Add best responses, or random if there were none
                if self._best_responses or mask.all():
                    mask[self._best_responses] = True
                else:
                    # This will happen if there's only one equilibria in the
                    # subgame, and it's a true equilibria
                    mask[np.random.choice(np.nonzero(~mask)[0])] = True

            eqa = (np.concatenate([eq[None] for eq in self._equilibria])
                   if self._equilibria
                   else np.empty((0, self._game.num_role_strats), float))
            return eqa, last_mask


class _InnerLoop(object):
    """Object to keep track of inner loop progress"""

    def __init__(self, sched, game, regret_thresh, dist_thresh, max_resamples,
                 subgame_size, num_equilibria, num_backups, devs_by_role):
        # Data
        self._sched = sched
        self._game = game

        # Parameters
        self._num_eq = num_equilibria
        self._num_backups = num_backups
        self._sub_size = subgame_size
        self._max_resamples = max_resamples
        self._regret_thresh = regret_thresh
        self._dist_thresh = dist_thresh
        self._init_role_dev = 0 if devs_by_role else None

        # Bookkeeping
        self._threads = queue.Queue()
        self._exp_subgames = collect.BitSet()
        self._exp_subgames_lock = threading.Lock()
        self._exp_mix = collect.MixtureSet(dist_thresh)
        self._exp_mix_lock = threading.Lock()
        self._backups = [queue.PriorityQueue() for _
                         in range(self._game.num_roles)]
        self._equilibria = []
        self._run_lock = threading.Lock()  # only one `run` at a time
        self._nash_lock = threading.BoundedSemaphore(3)  # only 3 `mixed_nash` at a time # noqa
        self._thread_failed = False

    def _add_subgame(self, sub_mask, count):
        if count > self._max_resamples:  # pragma: no cover
            _log.error("couldn't find equilibrium in subgame %s",
                       sub_mask.astype(int))
            return
        with self._exp_subgames_lock:
            schedule = count > 1 or self._exp_subgames.add(sub_mask)
        if schedule:
            _log.info('scheduling subgame %s%s', sub_mask.astype(int),
                      ' {:d}'.format(count) if count > 1 else '')
            thread = threading.Thread(
                target=lambda: self._run_subgame(sub_mask, count))
            thread.start()
            self._threads.put(thread)

    def _run_subgame(self, sub_mask, count):
        try:
            game = subgame.subgame(self._sched.get_subgame(sub_mask, count),
                                   sub_mask)
            with self._nash_lock:
                eqa = subgame.translate(nash.mixed_nash(
                    game, regret_thresh=self._regret_thresh,
                    dist_thresh=self._dist_thresh, processes=1), sub_mask)

            if eqa.size:
                for eqm in eqa:
                    self._add_deviations(eqm, self._init_role_dev)
            else:
                self._add_subgame(sub_mask, count + 1)  # pragma: no cover
        except Exception as ex:  # pragma: no cover
            self._thread_failed = True
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            raise ex

    def _add_deviations(self, mix, role_index):
        with self._exp_mix_lock:
            unseen = ((role_index is not None and role_index > 0)
                      or self._exp_mix.add(mix))
        if unseen:
            _log.info('scheduling deviations from %s with role %s', mix,
                      role_index)
            thread = threading.Thread(
                target=lambda: self._run_deviations(mix, role_index))
            thread.start()
            self._threads.put(thread)

    def _run_deviations(self, mix, role_index):
        try:
            support = mix > 0
            game = self._sched.get_deviations(support, 1, role_index)
            gains = regret.mixture_deviation_gains(game, mix)
            if role_index is None:
                assert not np.isnan(gains).any(), "There were nan regrets"
                if np.all(gains <= self._regret_thresh):  # Found equilibrium
                    _log.warning('found equilibrium %s with regret %f', mix,
                                 gains.max())
                    self._equilibria.append(mix)  # atomic
                else:
                    for ri, rgains in enumerate(game.role_split(gains)):
                        self._queue_subgames(support, rgains, ri)

            else:  # Set role index
                rgains = game.role_split(gains)[role_index]
                assert not np.isnan(rgains).any(), "There were nan regrets"
                if np.all(rgains <= self._regret_thresh):  # No deviations
                    role_index += 1
                    if role_index < game.num_roles:  # Explore next deviation
                        self._add_deviations(mix, role_index)
                    else:  # found equilibrium
                        _log.warning('found equilibrium %s with regret %f',
                                     mix, gains.max())
                        self._equilibria.append(mix)  # atomic
                else:
                    self._queue_subgames(support, rgains, role_index)
        except Exception as ex:  # pragma: no cover
            self._thread_failed = True
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            raise ex

    def _queue_subgames(self, support, role_gains, role_index):
        rs = self._game.role_starts[role_index]
        back = self._backups[role_index]

        # Handle best response
        br = np.argmax(role_gains)
        if (role_gains[br] > self._regret_thresh
                and support.sum() < self._sub_size):
            br_sub = support.copy()
            br_sub[rs + br] = True
            self._add_subgame(br_sub, 1)
        else:
            br = None  # Add best response to backup

        for si, gain in enumerate(role_gains):
            if si == br:
                continue
            sub = support.copy()
            sub[rs + si] = True
            back.put((-gain, id(sub), sub))  # id for tie-breaking

    def run(self):
        with self._run_lock:
            # Clear Data structures
            assert self._threads.empty()
            self._exp_subgames.clear()
            self._exp_mix.clear()
            for backup in self._backups:
                while not backup.empty():
                    # This only happens when run twice in a row
                    backup.get()  # pragma: no cover
            self._equilibria.clear()
            self._thread_failed = False

            # Quiesce first time
            for sub_mask in self._game.pure_subgames():
                self._add_subgame(sub_mask, 1)
            while not self._threads.empty():
                if self._thread_failed:
                    raise RuntimeError("a thread failed, checl log for details")  # pragma: no cover # noqa
                else:
                    self._threads.get().join()

            # Add backup games and quiesce again if less than desired number of
            # equilibria
            while (len(self._equilibria) < self._num_eq
                   and not all(q.empty() for q in self._backups)):
                for back in self._backups:
                    for _ in range(self._num_backups):
                        if back.empty():
                            break
                        self._add_subgame(back.get()[2], 1)
                while not self._threads.empty():
                    if self._thread_failed:
                        raise RuntimeError("a thread failed, checl log for details")  # pragma: no cover # noqa
                    else:
                        self._threads.get().join()

            # Return equilibria
            if self._equilibria:
                return np.concatenate([eq[None] for eq in self._equilibria])
            else:
                return np.empty((0, self._game.num_role_strats), float)  # pragma: no cover # noqa


class _Scheduler(object):
    """Scheduler abstraction that allows scheduling deviations and subgames as
    primitives"""

    def __init__(self, prof_sched, reduction):
        self._red = reduction
        self._prof_sched = prof_sched
        self._profiles = {}
        self._lock = threading.Lock()

    def _get_game(self, profiles, count):
        promises = []
        with self._lock:
            for prof in profiles:
                hprof = utils.hash_array(prof)
                prof = self._profiles.setdefault(
                    hprof, _PayoffData(self._prof_sched, prof))
                promise = prof.schedule(count)
                promises.append(promise)
        payoffs = np.concatenate([prom.get()[None] for prom in promises])
        return self._red.reduce_game(rsgame.game_copy(
            self._red.full_game, profiles, payoffs), True)

    def _subgame_profiles(self, subgame_mask):
        return self._red.expand_profiles(subgame.translate(
            subgame.subgame(self._red.red_game, subgame_mask).all_profiles(),
            subgame_mask))

    def get_subgame_scheduler(self, subgame_mask):
        num_strats = self._red.full_game.role_reduce(subgame_mask)
        reduction = self._red.__class__(num_strats,
                                        self._red.full_game.num_players,
                                        self._red.red_game.num_players)
        return _SubgameScheduler(self, reduction, subgame_mask)


class _SubgameScheduler(object):
    """Scheduler that schedules subgames"""

    def __init__(self, qsched, red, subgame_mask):
        self._qsched = qsched
        self._red = red
        self._submask = subgame_mask

    def get_subgame(self, subgame_mask, count):
        full_subgame = np.zeros(self._submask.size, bool)
        full_subgame[self._submask] = subgame_mask
        profiles = self._qsched._subgame_profiles(full_subgame)
        return subgame.subgame(self._qsched._get_game(profiles, count),
                               self._submask)

    def get_deviations(self, subgame_mask, count, role_index):
        full_subgame = np.zeros(self._submask.size, bool)
        full_subgame[self._submask] = subgame_mask
        subgame_profiles = self._qsched._subgame_profiles(full_subgame)
        dev_profiles = subgame.translate(
            self._red.expand_deviation_profiles(subgame_mask, role_index),
            self._submask)
        profiles = np.concatenate((subgame_profiles, dev_profiles))
        return subgame.subgame(self._qsched._get_game(profiles, count),
                               self._submask)


class _PayoffData(object):
    """Payoff data for a profile"""

    def __init__(self, prof_sched, profile):
        self._prof_sched = prof_sched
        self._profile = profile
        self._payoffs = np.zeros(profile.size, float)
        self._count = 0
        self._scheduled = 0

        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._promises = queue.Queue()
        self._thread_failed = False

    def _keep_going(self):
        with self._lock:
            return self._count < self._scheduled

    def _update(self):
        try:
            while self._keep_going():
                pay = self._promises.get().get()
                with self._lock:
                    self._count += 1
                    self._payoffs += (pay - self._payoffs) / self._count
                with self._cond:
                    self._cond.notify_all()
        except Exception as ex:  # pragma: no cover
            self._thread_failed = True
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            with self._cond:  # We must notify other threads that this failed
                self._cond.notify_all()
            raise ex

    def schedule(self, n):
        with self._lock:
            if self._count == self._scheduled and self._scheduled < n:
                threading.Thread(target=self._update).start()
            while self._scheduled < n:
                self._scheduled += 1
                self._promises.put(self._prof_sched.schedule(self._profile))
        return _PayoffPromise(self, n)


class _PayoffPromise(object):
    def __init__(self, pdata, n):
        self._pdata = pdata
        self._n = n

    def get(self):
        with self._pdata._cond:
            if self._pdata._thread_failed:
                raise RuntimeError("a thread failed, checl log for details")  # pragma: no cover # noqa
            else:
                self._pdata._cond.wait_for(
                    lambda: self._pdata._count >= self._n)
        return self._pdata._payoffs
