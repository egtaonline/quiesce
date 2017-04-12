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
        return _CountPromise(iter([self._sched.schedule(profile)
                                   for _ in range(self._count)]))

    def __enter__(self):
        self._sched.__enter__()
        return self

    def __exit__(self, *args):
        self._sched.__exit__(*args)


class _CountPromise(profsched.Promise):
    def __init__(self, proms):
        self._proms = proms
        self._value = None

    def get(self):
        if self._value is None:
            count = 1
            self._value = next(self._proms).get()
            for prom in self._proms:
                count += 1
                self._value += (prom.get() - self._value) / count
        return self._value
