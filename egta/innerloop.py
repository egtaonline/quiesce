import inspect
import logging
import queue
import threading
import time

import numpy as np
from gameanalysis import collect
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import restrict


_log = logging.getLogger(__name__)

# TODO There's something to be said about individual rationality constraints
# relative to best deviation, i.e. if all are positive gains, but negative
# payoff, that might mean we should warn, or at least explore something else.


def inner_loop(
        game, *, initial_restrictions=None, regret_thresh=1e-3,
        dist_thresh=0.1, support_thresh=1e-4, restricted_game_size=3,
        num_equilibria=1, num_backups=1, devs_by_role=False,
        at_least_one=False):
    """Inner loop a game using a scheduler

    Parameters
    ----------
    game : RsGame
        The game to find equilibria in. This function is most useful when game
        is a SchedulerGame, but any complete RsGame will work.
    initial_restriction : [[bool]], optional
        Initial restrictions to start inner loop from. If unspecified, every
        pure restriction is used.
    regret_thresh : float > 0, optional
        The maximum regret to consider an equilibrium an equilibrium.
    dist_thresh : float > 0, optional
        The minimum norm between two mixed profiles for them to be considered
        distinct.
    support_thresh : float > 0, optional
        Candidate equilibria strategies with probability lower than this will
        be truncated. This is useful because often Nash finding returns
        strategies with very low support, which now mean extra deviating
        strategies need to be sampled. Trimming saves these samples, but may
        increase regret above the threshold.
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
    initial_restrictions = (
        game.pure_restrictions() if initial_restrictions is None
        else np.asarray(initial_restrictions, bool))
    init_role_dev = 0 if devs_by_role else None

    threads = queue.Queue()
    exp_restrictions = collect.BitSet()
    exp_restrictions_lock = threading.Lock()
    backups = [queue.PriorityQueue() for _ in range(game.num_roles)]
    equilibria_lock = threading.Lock()
    equilibria = collect.mcces(dist_thresh)
    exceptions = []

    # Handle case where game might not have role_index key word
    if 'role_index' in inspect.signature(game.deviation_payoffs).parameters:
        def deviation_payoffs(mix, role_index):
            return game.deviation_payoffs(mix, role_index=role_index)
    else:
        def deviation_payoffs(mix, role_index):
            return game.deviation_payoffs(mix)

    def add_restriction(rest):
        with exp_restrictions_lock:
            schedule = exp_restrictions.add(rest)
        if schedule and not exceptions:
            thread = threading.Thread(
                target=run_restriction, args=(rest,), daemon=True)
            thread.start()
            threads.put(thread)
        return schedule

    def run_restriction(rest):
        try:
            _log.info(
                "scheduling profiles from restricted strategies %s",
                game.restriction_to_repr(rest))
            # TODO For anything but a SchedulerGame, this is a waste, but it
            # also potentially speeds up nash computation for Scheduler game.
            # The issue comes with computing the jacobian for noncomplete
            # mixtures. Scheduler games will return nan jacobians outside of
            # support for mixtures to save with sampling deviations.
            data = paygame.game_copy(game.restrict(rest))
            start = time.time()
            eqa = restrict.translate(data.trim_mixture_support(
                nash.mixed_nash(
                    data, regret_thresh=regret_thresh,
                    dist_thresh=dist_thresh,
                    at_least_one=at_least_one),
                thresh=support_thresh), rest)
            duration = time.time() - start
            if duration > 600:  # pragma: no cover
                _log.warning(
                    "equilibrium finding took %.0f seconds in "
                    "restricted game %s", duration,
                    game.restriction_to_repr(rest))
            if eqa.size:
                for eqm in eqa:
                    add_deviations(rest, eqm, init_role_dev)
            else:
                _log.warning(
                    "couldn't find equilibria in restricted game %s. This is "
                    "likely due to high variance in payoffs which means "
                    "quiesce should be re-run with more samples per profile. "
                    "This could also be fixed by performing a more expensive "
                    "equilibria search to always return one.",
                    game.restriction_to_repr(rest))
        except Exception as ex:  # pragma: no cover
            exceptions.append(ex)

    def add_deviations(rest, mix, role_index):
        if not exceptions:
            thread = threading.Thread(
                target=run_deviations, args=(rest, mix, role_index),
                daemon=True)
            thread.start()
            threads.put(thread)

    def run_deviations(rest, mix, role_index):
        # We need the restriction here, since trimming support may increase
        # regret of strategies in the initial restriction
        try:
            _log.info(
                "scheduling deviations from mixture %s%s",
                game.mixture_to_repr(mix),
                "" if role_index is None
                else " to role {}".format(game.role_names[role_index]))
            devs = deviation_payoffs(mix, role_index)
            exp = np.add.reduceat(devs * mix, game.role_starts)
            gains = devs - exp.repeat(game.num_role_strats)
            if role_index is None:
                if np.all((gains <= regret_thresh) | rest):
                    # Found equilibrium
                    reg = gains.max()
                    with equilibria_lock:
                        added = equilibria.add(mix, reg)
                    if added:
                        _log.warning('found equilibrium %s with regret %f',
                                     game.mixture_to_repr(mix), reg)
                else:
                    for ri, rgains in enumerate(np.split(
                            gains, game.role_starts[1:])):
                        queue_restrictions(rgains, ri, rest)

            else:  # Set role index
                rgains = np.split(gains, game.role_starts[1:])[role_index]
                rrest = np.split(rest, game.role_starts[1:])[role_index]
                if np.all((rgains <= regret_thresh) | rrest):  # No deviations
                    role_index += 1
                    if role_index < game.num_roles:  # Explore next deviation
                        add_deviations(rest, mix, role_index)
                    else:  # found equilibrium
                        # This should not require scheduling as to get here all
                        # deviations have to be scheduled
                        reg = regret.mixture_regret(game, mix)
                        _log.warning('found equilibrium %s with regret %f',
                                     game.mixture_to_repr(mix), reg)
                        with equilibria_lock:
                            equilibria.add(mix, reg)
                else:
                    queue_restrictions(rgains, role_index, rest)
        except Exception as ex:  # pragma: no cover
            exceptions.append(ex)

    def queue_restrictions(role_gains, role_index, rest):
        role_rest = np.split(rest, game.role_starts[1:])[role_index]
        if role_rest.all():
            return  # Can't deviate

        rest_size = rest.sum()
        rs = game.role_starts[role_index]

        br = np.nanargmax(np.where(role_rest, np.nan, role_gains))
        if (role_gains[br] > regret_thresh
                and rest_size < restricted_game_size):
            br_sub = rest.copy()
            br_sub[rs + br] = True
            add_restriction(br_sub)
        else:
            br = None  # Add best response to backup

        back = backups[role_index]
        for si, (gain, r) in enumerate(zip(role_gains, role_rest)):
            if si == br or r or gain <= 0:
                continue
            sub = rest.copy()
            sub[rs + si] = True
            # XXX Tie id to deterministic random source
            back.put((-gain, id(sub), sub))  # id for tie-breaking

    def join_threads():
        while not threads.empty():
            threads.get().join()
            if exceptions:
                raise exceptions[0]

    try:
        # Quiesce first time
        for rest in initial_restrictions:
            if np.all(np.add.reduceat(rest, game.role_starts) == 1):
                # Pure restriction, so we can skip right to deviations
                add_deviations(rest, rest.astype(float), init_role_dev)
            else:
                # Not pure, so equilibria are not obvious, schedule restriction
                # instead
                add_restriction(rest)
        join_threads()

        first_backups = True
        # Repeat with backups until found all
        while (len(equilibria) < num_equilibria
               and (not all(q.empty() for q in backups) or
                    not next(iter(exp_restrictions)).all())):
            if first_backups:
                _log.warning(
                    "scheduling backup restrictions. This only happens when "
                    "quiesce criteria could not be met with current maximum "
                    "restriction size (%d). This probably means that the "
                    "maximum restriction size should be increased. If this is "
                    "happening frequently, increasing the number of backups "
                    "taken at a time might be desired (currently %s).",
                    restricted_game_size, num_backups)
            else:
                _log.info("scheduling backup restrictions")
            first_backups = False

            for r, back in enumerate(backups):
                to_schedule = num_backups
                while to_schedule > 0:
                    # First try from backups
                    if not back.empty():
                        # This won't count if restriction already explored
                        to_schedule -= add_restriction(back.get()[-1])
                        continue
                    # Else pick unexplored subgames
                    rest = None
                    with exp_restrictions_lock:
                        for mask in exp_restrictions:
                            rmask = np.split(mask, game.role_starts[1:])[r]
                            if not rmask.all():
                                rest = mask.copy()
                                break
                    if rest is not None:
                        # TODO might be ideal if this is random, but it might
                        # make games not quiesce reliably
                        s = np.split(rest, game.role_starts[1:])[r].argmin()
                        rest[game.role_starts[r] + s] = True
                        add_restriction(rest)
                        to_schedule -= 1
                    else:
                        to_schedule = 0
            join_threads()

        # Return equilibria
        if equilibria:
            return np.stack([eqm for eqm, _ in equilibria])
        else:
            return np.empty((0, game.num_strats))  # pragma: no cover

    except Exception as ex:
        # Set exception so all threads exit
        exceptions.append(ex)
        raise ex
