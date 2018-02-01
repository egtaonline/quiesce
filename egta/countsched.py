import threading

import numpy as np

from egta import profsched


class CountScheduler(profsched.Scheduler):
    """A scheduler that wraps each profile in the mean of n profiles

    Parameters
    ----------
    sched : Scheduler
        The base scheduler that generates payoffs.
    count : int > 0
        The number of times a scheduler in the base scheduler should be sampled
        for each payoff returned by this scheduler.
    """

    def __init__(self, sched, count):
        assert count > 0, "count must be positive {:d}".format(count)
        self._sched = sched
        self._count = count

    def schedule(self, profile):
        return _CountPromise(
            [self._sched.schedule(profile) for _ in range(self._count)],
            np.zeros(self.game().num_strats))

    def game(self):
        return self._sched.game()

    def __enter__(self):
        self._sched.__enter__()
        return self

    def __exit__(self, *args):
        self._sched.__exit__(*args)


class _CountPromise(profsched.Promise):
    def __init__(self, proms, val):
        self._proms = proms
        self._value = val
        self._unset = True
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._unset:
                for i, prom in enumerate(self._proms, 1):
                    self._value += (prom.get() - self._value) / i
                self._value.setflags(write=False)
                self._unset = False
        return self._value
