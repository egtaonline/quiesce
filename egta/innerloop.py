import logging
import queue
import threading
import time
import warnings

import numpy as np
from gameanalysis import collect
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import restrict


_log = logging.getLogger(__name__)

# TODO There's something to be said about individual rationality constraints
# relative to best deviation, i.e. if all are positive gains, but negative
# payoff, that might mean we should warn, or at least explore something else.


def inner_loop(
        sched, *, initial_restrictions=None, regret_thresh=1e-3,
        dist_thresh=1e-2, max_resamples=10, restricted_game_size=3,
        num_equilibria=1, num_backups=1, devs_by_role=False,
        at_least_one=False):
    """Inner loop a game using a scheduler

    Parameters
    ----------
    sched : SparseScheduler
        The spare scheduler used to generate payoff data for profiles.
    initial_restriction : [[bool]], optional
        Initial restrictions to start inner loop from. If unspecified, every
        pure restriction is used.
    regret_thresh : float, optional
        The maximum regret to consider an equilibrium an equilibrium.
    dist_thresh : float, optional
        The minimum norm between two mixed profiles for them to be considered
        distinct.
    max_resamples : int > 0, optional
        The maximum number of times to resample a restricted game when no
        equilibria can be found before giving up.
    restricted_game_size : int > 0, optional
        The maximum restricted game support size with which beneficial
        deviations must be explored. Restricted games with support larger than
        this are queued and only explored in the event that no equilibrium can
        be found in beneficial deviations smaller than this.
    num_equilibria : int > 0, optional
        The number of equilibria to attempt to find. Only one is guaranteed,
        but this might be beneifical if the game has a known degenerate
        equilibria, but one which is still helpful as a deviating strategy.
    num_backups : int > 0, optional
        In the event that no equilibrium can be found in beneficial deviations
        to small restricted games, other restrictions will be explored. This
        parameter indicates how many restricted games for each role should be
        explored.
    devs_by_role : boolean, optional
        If specified, deviations will only be explored for each role in order,
        proceeding to the next role only when no beneficial deviations are
        found. This can reduce the number of profiles sampled, but may also
        fail to find certain equilibria due to the different path through
        restricted games.
    at_least_one : boolean, optional
        If specified, at least one equilibria will be found in each restricted
        game.  This has the potential to run for a very long time as it may
        take exponential time. If your regret threshold is not set too log for
        your game, this is relatively reasonable though.
    """
    game = sched.game()
    initial_restrictions = (
        game.pure_restrictions() if initial_restrictions is None
        else np.asarray(initial_restrictions, bool))
    init_role_dev = 0 if devs_by_role else None

    threads = queue.Queue()
    exp_restrictions = collect.BitSet()
    exp_restrictions_lock = threading.Lock()
    exp_mix = collect.MixtureSet(dist_thresh)
    exp_mix_lock = threading.Lock()
    backups = [queue.PriorityQueue() for _ in range(game.num_roles)]
    equilibria = []
    nash_lock = threading.Lock()  # nash is not thread safe
    exceptions = []

    def add_restriction(rest, count):
        if count > max_resamples:  # pragma: no cover
            _log.error("couldn't find equilibrium in restricted game %s",
                       game.restriction_to_repr(rest))
            return
        with exp_restrictions_lock:
            schedule = count > 1 or exp_restrictions.add(rest)
        if schedule and not exceptions:
            thread = threading.Thread(
                target=lambda: run_restriction(rest, count),
                daemon=True)
            thread.start()
            threads.put(thread)

    def run_restriction(rest, count):
        try:
            data = sched.get_restricted_game(rest, count).restrict(rest)
            with nash_lock:
                with warnings.catch_warnings():
                    # XXX For some reason, line-search in optimize throws a
                    # run-time warning when things get very small negative.
                    # This is potentially a error with the way we compute
                    # gradients, but it's not reproducible, so we ignore it.
                    warnings.simplefilter('ignore', RuntimeWarning)
                    start = time.time()
                    # FIXME Trim mixture support
                    eqa = restrict.translate(
                        nash.mixed_nash(
                            data, regret_thresh=regret_thresh,
                            dist_thresh=dist_thresh,
                            at_least_one=at_least_one),
                        rest)
                    duration = time.time() - start
                    if duration > 600:  # pragma: no cover
                        _log.warning(
                            'equilibrium finding took %.0f seconds in '
                            'restricted game %s', duration,
                            game.restriction_to_repr(rest))
            if eqa.size:
                for eqm in eqa:
                    add_deviations(eqm, init_role_dev)
            else:
                add_restriction(rest, count + 1)  # pragma: no cover
        except Exception as ex:  # pragma: no cover
            exceptions.append(ex)

    def add_deviations(mix, role_index):
        with exp_mix_lock:
            unseen = ((role_index is not None and role_index > 0)
                      or exp_mix.add(mix))
        if unseen and not exceptions:
            thread = threading.Thread(
                target=lambda: run_deviations(mix, role_index))
            thread.start()
            threads.put(thread)

    def run_deviations(mix, role_index):
        try:
            support = mix > 0
            data = sched.get_deviations(support, 1, role_index)
            gains = regret.mixture_deviation_gains(data, mix)
            if role_index is None:
                if np.all(gains <= regret_thresh):  # Found equilibrium
                    _log.warning('found equilibrium %s with regret %f',
                                 game.mixture_to_repr(mix),
                                 gains.max())
                    equilibria.append(mix)  # atomic
                else:
                    for ri, rgains in enumerate(np.split(
                            gains, game.role_starts[1:])):
                        queue_restrictions(support, rgains, ri)

            else:  # Set role index
                rgains = np.split(gains, game.role_starts[1:])[role_index]
                if np.all(rgains <= regret_thresh):  # No deviations
                    role_index += 1
                    if role_index < game.num_roles:  # Explore next deviation
                        add_deviations(mix, role_index)
                    else:  # found equilibrium
                        _log.warning('found equilibrium %s with regret %f',
                                     game.mixture_to_repr(mix),
                                     gains.max())
                        equilibria.append(mix)  # atomic
                else:
                    queue_restrictions(support, rgains, role_index)
        except Exception as ex:  # pragma: no cover
            exceptions.append(ex)

    def queue_restrictions(support, role_gains, role_index):
        rs = game.role_starts[role_index]
        back = backups[role_index]

        # Handle best response
        br = np.argmax(role_gains)
        if (role_gains[br] > regret_thresh
                and support.sum() < restricted_game_size):
            br_sub = support.copy()
            br_sub[rs + br] = True
            add_restriction(br_sub, 1)
        else:
            br = None  # Add best response to backup

        for si, gain in enumerate(role_gains):
            if si == br:
                continue
            sub = support.copy()
            sub[rs + si] = True
            back.put((-gain, id(sub), sub))  # id for tie-breaking

    def join_threads():
        try:
            while True:
                threads.get_nowait().join()
                if exceptions:
                    raise exceptions[0]
        except queue.Empty:
            pass

    try:
        # Quiesce first time
        for sub in initial_restrictions:
            if np.all(np.add.reduceat(sub, game.role_starts) == 1):
                # Pure restriction, so we can skip right to deviations
                add_deviations(sub.astype(float), init_role_dev)
            else:
                # Not pure, so equilibria are not obvious, schedule restriction
                # instead
                add_restriction(sub, 1)
        join_threads()

        # Repeat with backups until found all
        while (len(equilibria) < num_equilibria
               and not all(q.empty() for q in backups)):
            for back in backups:
                for _ in range(num_backups):
                    if back.empty():
                        break  # pragma: no cover
                    add_restriction(back.get()[2], 1)
            join_threads()

        # Return equilibria
        if equilibria:
            return np.stack(equilibria)
        else:
            return np.empty((0, game.num_strats))  # pragma: no cover

    except Exception as ex:
        # Set exception so all threads exit
        exceptions.append(ex)
        raise ex
