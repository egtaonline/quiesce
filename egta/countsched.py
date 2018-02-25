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

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._sched.__aexit__(*args)
