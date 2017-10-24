import collections
import logging
import queue
import threading
import warnings

import numpy as np
from gameanalysis import collect
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils
from gameanalysis.reduction import identity as idr


_log = logging.getLogger(__name__)


def inner_loop(prof_sched, game, red=idr, red_players=None, *,
               regret_thresh=1e-3, dist_thresh=1e-3, max_resamples=10,
               subgame_size=3, num_equilibria=1, num_backups=1,
               devs_by_role=False):
    """Inner loop a game using a scheduler

    Parameters
    ----------
    prof_sched : Scheduler
        The scheduler used to generate payoff data for profiles.
    game : RsGame
        The gameanalysis basegame for the game to find equilibria on.
    red : reduction, optional
        The reduction to apply to the game before scheduling. Using a reduction
        will sample fewer profiles, while only approximating equilibria. This
        is any of the valid `classes` in gameanalysis.reduction.
    red_players : ndarray-like, optional
        The amount of players to reduce the game to. This is required for
        reductions that require them, but can be left out for identity and
        twins reductions.
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
    return _InnerLoop(prof_sched, rsgame.emptygame_copy(game), red,
                      np.broadcast_to(np.asarray(red_players), game.num_roles)
                      if red_players is not None else None,
                      regret_thresh=regret_thresh, dist_thresh=dist_thresh,
                      max_resamples=max_resamples, subgame_size=subgame_size,
                      num_equilibria=num_equilibria, num_backups=num_backups,
                      devs_by_role=devs_by_role).run()


class _InnerLoop(object):
    """Object to keep track of inner loop progress"""

    def __init__(self, sched, game, red, red_players, regret_thresh,
                 dist_thresh, max_resamples, subgame_size, num_equilibria,
                 num_backups, devs_by_role):
        # Data
        self._sched = _Scheduler(sched, game, red, red_players)
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
        self._nash_lock = threading.Lock()  # nash is not thread safe
        self._exception = None

    def _add_subgame(self, sub_mask, count):
        if count > self._max_resamples:  # pragma: no cover
            _log.error("couldn't find equilibrium in subgame %s",
                       self._game.to_subgame_repr(sub_mask))
            return
        with self._exp_subgames_lock:
            schedule = count > 1 or self._exp_subgames.add(sub_mask)
        if schedule:
            _log.info('scheduling subgame %s%s',
                      self._game.to_subgame_repr(sub_mask),
                      ' {:d}'.format(count) if count > 1 else '')
            thread = threading.Thread(
                target=lambda: self._run_subgame(sub_mask, count))
            thread.start()
            self._threads.put(thread)

    def _run_subgame(self, sub_mask, count):
        if self._exception is not None:
            return  # Something failed
        try:
            game = self._sched.get_subgame(sub_mask, count).subgame(sub_mask)
            with self._nash_lock:
                with warnings.catch_warnings():
                    # XXX For some reason, linesearch in optimize throws a
                    # run-time warning when things get very small negative.
                    # This is potentially a error with the way we compute
                    # gradients, but it's not reproducible, so we ignore it.
                    warnings.simplefilter('ignore', RuntimeWarning)
                    eqa = subgame.translate(
                        nash.mixed_nash(
                            game, regret_thresh=self._regret_thresh,
                            dist_thresh=self._dist_thresh, processes=1),
                        sub_mask)

            if eqa.size:
                for eqm in eqa:
                    self._add_deviations(eqm, self._init_role_dev)
            else:
                self._add_subgame(sub_mask, count + 1)  # pragma: no cover
        except Exception as ex:  # pragma: no cover
            self._exception = ex

    def _add_deviations(self, mix, role_index):
        with self._exp_mix_lock:
            unseen = ((role_index is not None and role_index > 0)
                      or self._exp_mix.add(mix))
        if unseen:
            _log.info(
                'scheduling deviations from %s%s', self._game.to_mix_repr(mix),
                '' if role_index is None else ' with role {}'.format(
                    self._game.role_names[role_index]))
            thread = threading.Thread(
                target=lambda: self._run_deviations(mix, role_index))
            thread.start()
            self._threads.put(thread)

    def _run_deviations(self, mix, role_index):
        if self._exception is not None:
            return  # Something failed
        try:
            support = mix > 0
            game = self._sched.get_deviations(support, 1, role_index)
            gains = regret.mixture_deviation_gains(game, mix)
            if role_index is None:
                assert not np.isnan(gains).any(), "There were nan regrets"
                if np.all(gains <= self._regret_thresh):  # Found equilibrium
                    _log.warning('found equilibrium %s with regret %f',
                                 self._game.to_mix_repr(mix),
                                 gains.max())
                    self._equilibria.append(mix)  # atomic
                else:
                    for ri, rgains in enumerate(np.split(
                            gains, game.role_starts[1:])):
                        self._queue_subgames(support, rgains, ri)

            else:  # Set role index
                rgains = np.split(gains, game.role_starts[1:])[role_index]
                assert not np.isnan(rgains).any(), "There were nan regrets"
                if np.all(rgains <= self._regret_thresh):  # No deviations
                    role_index += 1
                    if role_index < game.num_roles:  # Explore next deviation
                        self._add_deviations(mix, role_index)
                    else:  # found equilibrium
                        _log.warning('found equilibrium %s with regret %f',
                                     self._game.to_mix_repr(mix), gains.max())
                        self._equilibria.append(mix)  # atomic
                else:
                    self._queue_subgames(support, rgains, role_index)
        except Exception as ex:  # pragma: no cover
            self._exception = ex

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
        try:
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
                    self._threads.get().join()
                if self._exception is not None:
                    raise self._exception

                # Add backup games and quiesce again if less than desired
                # number of equilibria
                while (len(self._equilibria) < self._num_eq
                       and not all(q.empty() for q in self._backups)):
                    for back in self._backups:
                        for _ in range(self._num_backups):
                            if back.empty():
                                break  # pragma: no cover
                            self._add_subgame(back.get()[2], 1)
                    while not self._threads.empty():
                        self._threads.get().join()
                    if self._exception is not None:
                        raise self._exception

                # Return equilibria
                if self._equilibria:
                    return np.concatenate(
                        [eq[None] for eq in self._equilibria])
                else:
                    return np.empty((0, self._game.num_role_strats), float)  # pragma: no cover # noqa

        except Exception as ex:
            # Set exception so all threads exit
            self._exception = ex
            # Kill scheduler
            self._sched._prof_sched.__exit__()
            # Wait for all threads to finish now that all scheduler promises
            # should have returned. This should only happen if an exception is
            # thrown internally i.e. not by the scheduler.
            while not self._threads.empty():
                self._threads.get().join()
            raise ex


class _Scheduler(object):
    """Scheduler abstraction

    This abstraction supports scheduling deviations and subgames as
    primitives"""

    def __init__(self, prof_sched, game, red, red_players):
        self._game = game
        self._red_game = red.reduce_game(game, red_players)
        self._red = red
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
        return self._red.reduce_game(paygame.game_replace(
            self._game, profiles, payoffs), self._red_game.num_role_players)

    def _subgame_profiles(self, subgame_mask):
        return self._red.expand_profiles(self._game, subgame.translate(
            self._red_game.subgame(subgame_mask).all_profiles(), subgame_mask))

    def get_subgame(self, subgame_mask, count):
        return self._get_game(self._subgame_profiles(subgame_mask), count)

    def get_deviations(self, subgame_mask, count, role_index):
        subgame_profiles = self._subgame_profiles(subgame_mask)
        dev_profiles = self._red.expand_deviation_profiles(
            self._game, subgame_mask, self._red_game.num_role_players,
            role_index)
        return self._get_game(np.concatenate((subgame_profiles, dev_profiles)),
                              count)


class _PayoffData(object):
    """Payoff data for a profile"""

    def __init__(self, prof_sched, profile):
        self._prof_sched = prof_sched
        self._profile = profile
        self._payoffs = np.zeros(profile.size, float)
        self._count = 0

        self._lock = threading.Lock()
        self._promises = collections.deque()

    def schedule(self, n):
        with self._lock:
            for _ in range(n - self._count - len(self._promises)):
                prom = self._prof_sched.schedule(self._profile)
                self._promises.append(_UpdatePromise(self, prom))
        return _PayoffPromise(self, n)


class _UpdatePromise(object):
    def __init__(self, data, promise):
        self._data = data
        self._promise = promise
        self._lock = threading.Lock()
        self._updated = False

    def wait(self):
        with self._lock:
            if not self._updated:
                pay = self._promise.get()
                with self._data._lock:
                    assert self._data._promises.popleft() == self
                    self._data._count += 1
                    self._data._payoffs += (
                        (pay - self._data._payoffs) / self._data._count)
                self._updated = True


class _PayoffPromise(object):
    def __init__(self, data, n):
        self._data = data
        self._n = n

    def get(self):
        while True:
            with self._data._lock:
                assert self._n <= self._data._count + len(self._data._promises)
                if self._n <= self._data._count:
                    return self._data._payoffs
                updater = self._data._promises[0]
            updater.wait()
