import asyncio

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

    async def sample_payoffs(self, profile):
        payoffs = await asyncio.gather(*[
            self._sched.sample_payoffs(profile) for _ in range(self._count)])
        payoff = np.zeros(self.game().num_strats)
        for i, pay in enumerate(payoffs, 1):
            payoff += (pay - payoff) / i
        return payoff

    def game(self):
        return self._sched.game()

    def __enter__(self):
        self._sched.__enter__()
        return self

    def __exit__(self, *args):
        self._sched.__exit__(*args)
