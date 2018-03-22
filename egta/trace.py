import asyncio
import functools
import itertools
import logging

import numpy as np
from scipy import interpolate
from scipy.sparse import csgraph
from gameanalysis import rsgame
from gameanalysis import trace

from egta import asyncgame
from egta import innerloop


# TODO Expose max step
async def trace_all_equilibria(
        agame0, agame1, *, regret_thresh=1e-3, dist_thresh=0.1, executor=None,
        **innerloop_args):
    """Trace out all equilibria between all games

    Parameters
    ----------
    agame0 : AsyncGame
        The game that is played when time is 0.
    agame1 : AsyncGame
        The game that is played when time is 1.
    regret_thresh : float, optional
        The threshold for epsilon regret for equilibria returned.
    exectutor : Executor, optional
        The executor to run computation intensive operations in.
    """
    assert rsgame.emptygame_copy(agame0) == rsgame.emptygame_copy(agame1)
    loop = asyncio.get_event_loop()
    trace_args = dict(regret_thresh=regret_thresh)
    innerloop_args.update(
        trace_args, executor=executor, dist_thresh=dist_thresh)

    async def trace_eqm(eqm, t):
        supp = eqm > 0
        game0, game1 = await asyncio.gather(
            agame0.get_deviation_game(supp),
            agame1.get_deviation_game(supp))
        return await loop.run_in_executor(
            executor, functools.partial(
                trace.trace_equilibria, game0, game1, t, eqm, **trace_args))

    async def trace_between(lower, upper):
        if upper <= lower:
            return ()
        mid = (lower + upper) / 2

        midgame = asyncgame.merge(agame0, agame1, mid)
        eqa = await innerloop.inner_loop(midgame, **innerloop_args)

        if not eqa.size:  # pragma: no cover
            logging.warning("found no equilibria in %s", midgame)
            return ()

        # XXX This shouldn't really be async as all data should be there, could
        # potentially add a get_nowait to async game
        traces = await asyncio.gather(*[
            trace_eqm(eqm, mid) for eqm in eqa])

        lupper = min(t[0] for t, _ in traces)
        ulower = max(t[-1] for t, _ in traces)
        logging.warning(
            "traced %s out to %g - %g", midgame, lupper, ulower)

        lower_traces, upper_traces = await asyncio.gather(
            trace_between(lower, lupper), trace_between(ulower, upper))
        # Lazily extend them
        return itertools.chain(lower_traces, traces, upper_traces)

    traces = list(await trace_between(0.0, 1.0))
    traces = _merge_traces(traces, dist_thresh, 'linear')
    return sorted(traces, key=lambda tr: (tr[0][0], tr[0][-1]))


def _trace_distance(trace1, trace2, interp):
    """Compute the distance between traces

    This uses interpolation to estimate each trace at arbitrary points in time
    and then computes the average time-weighted norm between the traces."""
    t1, eqa1 = trace1
    t2, eqa2 = trace2
    if t1[-1] < t2[0] or t2[-1] < t1[0]:
        return np.inf

    tmin = max(t1[0], t2[0])
    tmax = min(t1[-1], t2[-1])

    # XXX This sorted merge could be more efficient
    ts = np.concatenate([
        t1[(tmin <= t1) & (t1 <= tmax)], t2[(tmin <= t2) & (t2 <= tmax)]])
    ts.sort()

    eqa1i = interpolate.interp1d(t1, eqa1, interp, 0)(ts)
    eqa2i = interpolate.interp1d(t2, eqa2, interp, 0)(ts)
    errs = np.linalg.norm(eqa1i - eqa2i, axis=1)
    return np.diff(ts).dot(errs[:-1] + errs[1:]) / 2 / (tmax - tmin)


def _merge_traces(traces, thresh, interp):
    """Merge a list of traces

    Parameters
    ----------
    traces : [(ts, eqa)]
        A list of traces, which are themselves tuples of times and equilibria.
    thresh : float
        How similar traces need to be in order to be merged. This is the
        average norm between the traces.
    interp : str
        The way to interpolate between equilibria. This is passed to interp1d.
    """
    distances = np.zeros([len(traces)] * 2)
    for i, trace1 in enumerate(traces):
        for j, trace2 in enumerate(traces[:i]):
            distances[i, j] = distances[j, i] = _trace_distance(
                trace1, trace2, interp)
    num, comps = csgraph.connected_components(distances <= thresh, False)
    new_traces = []
    for i in range(num):
        ts = np.concatenate([
            t for (t, _), m in zip(traces, comps == i) if m])
        eqa = np.concatenate([
            eqms for (_, eqms), m in zip(traces, comps == i) if m])
        inds = np.argsort(ts)
        new_traces.append((ts[inds], eqa[inds]))
    # FIXME These merged traces are noisy due to bounds in regret. Ideally we'd
    # do some smoothing or something... locally weighted linear regression?
    return new_traces
