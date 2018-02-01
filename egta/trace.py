import logging
import queue
import threading

from gameanalysis import merge
from gameanalysis import rsgame

from egta import innerloop


_log = logging.getLogger(__name__)


# TODO Expose max step, make regret_thresh an argument and append it to
# innerloop_args
def trace_equilibria(game1, game2, **innerloop_args):
    assert rsgame.emptygame_copy(game1) == rsgame.emptygame_copy(game1)
    trace_args = {
        'regret_thresh': innerloop_args.get('regret_thresh', 1e-3) / 10}

    threads = queue.Queue()

    def trace_eqm(eqm, t, res):
        """Trace an equilibrium in both directions"""
        def run():
            _log.info("tracing equilibrium %s from ratio %g",
                      game1.mixture_to_repr(eqm), t)
            res.append(merge.trace_equilibria(
                game1, game2, t, eqm, **trace_args))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    # First find initial equilibria in each game
    def init_eqa(game, t, res):
        """Find the initial equilibria"""
        def run():
            eqa = innerloop.inner_loop(game, **innerloop_args)
            if not eqa.size:  # pragma: no cover
                _log.warning("found no equilibria for t: %g", t)
            for eqm in eqa:
                threads.put(trace_eqm(eqm, t, res))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.put(thread)

    traces1 = []
    traces2 = []

    init_eqa(game1, 0, traces1)
    init_eqa(game2, 1, traces2)

    while not threads.empty():
        threads.get().join()

    # Now take the mid region and continue to trace outward from the midpoint
    # of an unknown region, and then continue tracing if there are unexplored
    # regions.
    traces = []
    traces.extend(traces1)
    traces.extend(traces2)

    def init_mid(lower, upper):
        if upper < lower:
            return
        t = (lower + upper) / 2

        def run():
            eqa = innerloop.inner_loop(
                merge.merge(game1, game2, t), **innerloop_args)
            if not eqa.size:  # pragma: no cover
                _log.warning("found no equilibria for t: %g", t)
                return

            trs = []
            thrds = queue.Queue()
            for eqm in eqa:
                thrds.put(trace_eqm(eqm, t, trs))
            while not thrds.empty():
                thrds.get().join()
            traces.extend(trs)

            lupper = min(t[0] for t, _ in trs)
            ulower = max(t[-1] for t, _ in trs)
            _log.info("traced %g out to %g - %g", t, lupper, ulower)

            init_mid(lower, lupper)
            init_mid(ulower, upper)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.put(thread)

    init_mid(max((t[-1] for t, _ in traces1), default=0),
             min((t[0] for t, _ in traces2), default=1))
    while not threads.empty():
        threads.get().join()

    return sorted(traces, key=lambda tr: (tr[0][0], tr[0][-1]))
