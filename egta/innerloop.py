import asyncio
import functools
import heapq
import logging

import numpy as np
from gameanalysis import collect
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import restrict


# TODO There's something to be said about individual rationality constraints
# relative to best deviation, i.e. if all are positive gains, but negative
# payoff, that might mean we should warn, or at least explore something else.


async def inner_loop(
        agame, *, initial_restrictions=None, regret_thresh=1e-3,
        dist_thresh=0.1, support_thresh=1e-4, restricted_game_size=3,
        num_equilibria=1, num_backups=1, devs_by_role=False,
        at_least_one=False, executor=None):
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
    executor : Executor, optional
        The executor to be used for Nash finding. The default setting will
        allow async networking calls to continue to happen during long nash
        finding, but buy using a process pool this can take advantage of
        parallel computation.
    """
    init_role_dev = 0 if devs_by_role else None

    exp_restrictions = collect.bitset()
    exp_restrictions.add(np.zeros(agame.num_strats, bool))
    backups = [[] for _ in range(agame.num_roles)]
    equilibria = collect.mcces(dist_thresh)
    loop = asyncio.get_event_loop()

    async def add_restriction(rest):
        if not exp_restrictions.add(rest):
            return  # already explored
        if agame.is_pure_restriction(rest):
            # Short circuit for pure restriction
            await add_deviations(rest, rest.astype(float), init_role_dev)
            return
        data = await agame.get_restricted_game(rest)
        reqa = await loop.run_in_executor(executor, functools.partial(
            nash.mixed_nash, data, regret_thresh=regret_thresh,
            dist_thresh=dist_thresh, at_least_one=at_least_one))
        if reqa.size:
            eqa = restrict.translate(data.trim_mixture_support(reqa), rest)
            await asyncio.gather(*[
                add_deviations(rest, eqm, init_role_dev) for eqm in eqa])
        else:
            logging.warning(
                "%s: couldn't find equilibria in restricted game %s. This is "
                "likely due to high variance in payoffs which means "
                "quiesce should be re-run with more samples per profile. "
                "This could also be fixed by performing a more expensive "
                "equilibria search to always return one.",
                agame, agame.restriction_to_repr(rest))

    async def add_deviations(rest, mix, role_index):
        # We need the restriction here, since trimming support may increase
        # regret of strategies in the initial restriction
        data = await agame.get_deviation_game(mix > 0, role_index)
        devs = data.deviation_payoffs(mix)
        exp = np.add.reduceat(devs * mix, agame.role_starts)
        gains = devs - exp.repeat(agame.num_role_strats)
        if role_index is None:
            if np.all((gains <= regret_thresh) | rest):
                # Found equilibrium
                reg = gains.max()
                if equilibria.add(mix, reg):
                    logging.warning(
                        "%s: found equilibrium %s with regret %f", agame,
                        agame.mixture_to_repr(mix), reg)
            else:
                await asyncio.gather(*[
                    queue_restrictions(rgains, ri, rest)
                    for ri, rgains in enumerate(np.split(
                            gains, agame.role_starts[1:]))])

        else:  # Set role index
            rgains = np.split(gains, agame.role_starts[1:])[role_index]
            rrest = np.split(rest, agame.role_starts[1:])[role_index]
            if np.all((rgains <= regret_thresh) | rrest):  # No deviations
                role_index += 1
                if role_index < agame.num_roles:  # Explore next deviation
                    await add_deviations(rest, mix, role_index)
                else:  # found equilibrium
                    # This should not require scheduling as to get here all
                    # deviations have to be scheduled
                    data = await agame.get_deviation_game(mix > 0)
                    reg = regret.mixture_regret(data, mix)
                    if equilibria.add(mix, reg):
                        logging.warning(
                            "%s: found equilibrium %s with regret %f", agame,
                            agame.mixture_to_repr(mix), reg)
            else:
                await queue_restrictions(rgains, role_index, rest)

    async def queue_restrictions(role_gains, role_index, rest):
        role_rest = np.split(rest, agame.role_starts[1:])[role_index]
        if role_rest.all():
            return  # Can't deviate

        rest_size = rest.sum()
        rs = agame.role_starts[role_index]

        br = np.nanargmax(np.where(role_rest, np.nan, role_gains))
        if (role_gains[br] > regret_thresh
                and rest_size < restricted_game_size):
            br_sub = rest.copy()
            br_sub[rs + br] = True
            await add_restriction(br_sub)
        else:
            br = None  # Add best response to backup

        back = backups[role_index]
        for si, (gain, r) in enumerate(zip(role_gains, role_rest)):
            if si == br or r or gain <= 0:
                continue
            sub = rest.copy()
            sub[rs + si] = True
            # XXX Tie id to deterministic random source
            heapq.heappush(back, (-gain, id(sub), sub))  # id for tie-breaking

    restrictions = (
        agame.pure_restrictions() if initial_restrictions is None
        else np.asarray(initial_restrictions, bool))

    iteration = 0
    while (len(equilibria) < num_equilibria
           and (any(q for q in backups) or
                not next(iter(exp_restrictions)).all())):
        if iteration == 1:
            logging.warning(
                "%s: scheduling backup restrictions. This only happens "
                "when quiesce criteria could not be met with current "
                "maximum restriction size (%d). This probably means that "
                "the maximum restriction size should be increased. If "
                "this is happening frequently, increasing the number of "
                "backups taken at a time might be desired (currently %s).",
                agame, restricted_game_size, num_backups)
        elif iteration > 1:
            logging.info("%s: scheduling backup restrictions", agame)

        await asyncio.gather(*[
            add_restriction(r) for r in restrictions])

        restrictions = collect.bitset(exp_restrictions)
        for r, back in enumerate(backups):
            unscheduled = num_backups
            while unscheduled > 0 and back:
                rest = heapq.heappop(back)[-1]
                unscheduled -= restrictions.add(rest)
            for _ in range(unscheduled):
                added = False
                for mask in restrictions:
                    rmask = np.split(mask, agame.role_starts[1:])[r]
                    if rmask.all():
                        continue
                    rest = mask.copy()
                    # TODO We could randomize instead of taking the first
                    # strategy, but this would remove reproducability
                    s = np.split(rest, agame.role_starts[1:])[r].argmin()
                    rest[agame.role_starts[r] + s] = True
                    restrictions.add(rest)
                    added = True
                    break
                if not added:
                    break
        iteration += 1

    # Return equilibria
    if equilibria:
        return np.stack([eqm for eqm, _ in equilibria])
    else:
        return np.empty((0, agame.num_strats))  # pragma: no cover
