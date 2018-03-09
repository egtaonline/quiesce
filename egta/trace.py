import asyncio
import itertools
import logging

from gameanalysis import rsgame
from gameanalysis import trace

from egta import asyncgame
from egta import innerloop


# TODO Expose max step
async def trace_all_equilibria(
        agame0, agame1, regret_thresh=1e-3, **innerloop_args):
    assert rsgame.emptygame_copy(agame0) == rsgame.emptygame_copy(agame1)
    trace_args = dict(regret_thresh=regret_thresh)
    innerloop_args.update(trace_args)

    async def trace_eqm(eqm, t):
        supp = eqm > 0
        game0, game1 = await asyncio.gather(
            agame0.get_deviation_game(supp),
            agame1.get_deviation_game(supp))
        # TODO Should this be wrapped in an executor?
        return trace.trace_equilibria(
            game0, game1, t, eqm, **trace_args)

    async def trace_between(lower, upper):
        if upper <= lower:
            return ()
        mid = (lower + upper) / 2

        midgame = asyncgame.merge(agame0, agame1, mid)
        eqa = await innerloop.inner_loop(midgame, **innerloop_args)

        if not eqa.size:  # pragma: no cover
            logging.warning("%s: found no equilibria for t: %g", midgame, mid)
            return ()

        # XXX This shouldn't really be async as all data should be there, could
        # potentially add a get_nowait to async game
        traces = await asyncio.gather(*[
            trace_eqm(eqm, mid) for eqm in eqa])

        lupper = min(t[0] for t, _ in traces)
        ulower = max(t[-1] for t, _ in traces)
        logging.warning(
            "%s: traced %g out to %g - %g", midgame, mid, lupper, ulower)

        lower_traces, upper_traces = await asyncio.gather(
            trace_between(lower, lupper), trace_between(ulower, upper))
        # Lazily extend them
        return itertools.chain(lower_traces, traces, upper_traces)

    traces = await trace_between(0.0, 1.0)
    return sorted(traces, key=lambda tr: (tr[0][0], tr[0][-1]))
