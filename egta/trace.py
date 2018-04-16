"""Module for tracing continuous equilibria"""
import asyncio
import functools
import itertools
import logging

import numpy as np
from scipy.sparse import csgraph
from gameanalysis import rsgame
from gameanalysis import trace
from gameanalysis import utils

from egta import asyncgame
from egta import innerloop


async def trace_all_equilibria(
        agame0, agame1, *, regret_thresh=1e-3, dist_thresh=0.1, max_step=0.1,
        executor=None, **innerloop_args):
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
    utils.check(
        rsgame.empty_copy(agame0) == rsgame.empty_copy(agame1),
        'games must have same structure')
    loop = asyncio.get_event_loop()
    trace_args = dict(regret_thresh=regret_thresh, max_step=max_step)
    innerloop_args.update(
        executor=executor, regret_thresh=regret_thresh,
        dist_thresh=dist_thresh)

    async def trace_eqm(eqm, prob):
        """Trace and equilibrium out from prob"""
        game0 = agame0.get_game()
        game1 = agame1.get_game()
        (pr0, eqa0), (pr1, eqa1) = await asyncio.gather(
            loop.run_in_executor(executor, functools.partial(
                trace.trace_equilibrium, game0, game1, prob, eqm, 0,
                **trace_args)),
            loop.run_in_executor(executor, functools.partial(
                trace.trace_equilibrium, game0, game1, prob, eqm, 1,
                **trace_args)))
        return (
            np.concatenate([pr0[::-1], pr1[1:]]),
            np.concatenate([eqa0[::-1], eqa1[1:]]))

    async def trace_between(lower, upper):
        """Trace between times lower and upper"""
        if upper <= lower:
            return ()
        mid = (lower + upper) / 2

        midgame = asyncgame.mix(agame0, agame1, mid)
        eqa = await innerloop.inner_loop(midgame, **innerloop_args)

        if not eqa.size:  # pragma: no cover
            logging.warning('found no equilibria in %s', midgame)
            return ()

        traces = await asyncio.gather(*[
            trace_eqm(eqm, mid) for eqm in eqa])

        lupper = min(t[0] for t, _ in traces)
        ulower = max(t[-1] for t, _ in traces)
        logging.warning(
            'traced %s out to %g - %g', midgame, lupper, ulower)

        lower_traces, upper_traces = await asyncio.gather(
            trace_between(lower, lupper), trace_between(ulower, upper))
        # Lazily extend them
        return itertools.chain(lower_traces, traces, upper_traces)

    traces = list(await trace_between(0.0, 1.0))
    traces = _merge_traces(
        agame0.get_game(), agame1.get_game(), traces, dist_thresh, trace_args)
    # FIXME Smooth these out by interpolating between eqa on either side and
    # keeping new value if it has lower regret
    return sorted(traces, key=lambda tr: (tr[0][0], tr[0][-1]))


def _trace_distance(game0, game1, trace1, trace2, trace_args):
    """Compute the distance between traces

    This uses interpolation to estimate each trace at arbitrary points in time
    and then computes the average time-weighted norm between the traces."""
    time1, eqa1 = trace1
    time2, eqa2 = trace2
    if time1[-1] < time2[0] or time2[-1] < time1[0]:
        return np.inf

    tmin = max(time1[0], time2[0])
    tmax = min(time1[-1], time2[-1])

    # XXX This sorted merge could be more efficient
    times = np.concatenate([
        time1[(tmin <= time1) & (time1 <= tmax)],
        time2[(tmin <= time2) & (time2 <= tmax)]])
    times.sort()

    eqa1i = trace.trace_interpolate(
        game0, game1, time1, eqa1, times, **trace_args)
    eqa2i = trace.trace_interpolate(
        game0, game1, time2, eqa2, times, **trace_args)
    errs = np.linalg.norm(eqa1i - eqa2i, axis=1)
    return np.diff(times).dot(errs[:-1] + errs[1:]) / 2 / (tmax - tmin)


def _merge_traces(game0, game1, traces, thresh, trace_args): # pylint: disable=too-many-locals
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
                game0, game1, trace1, trace2, trace_args)
    num, comps = csgraph.connected_components(distances <= thresh, False)
    new_traces = []
    for i in range(num):
        times = np.concatenate([
            t for (t, _), m in zip(traces, comps == i) if m])
        eqa = np.concatenate([
            eqms for (_, eqms), m in zip(traces, comps == i) if m])
        inds = np.argsort(times)
        new_traces.append((times[inds], eqa[inds]))
    return new_traces
