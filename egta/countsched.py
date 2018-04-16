"""Module for a scheduler that summarizes several samples"""
import asyncio

import numpy as np
from gameanalysis import utils

from egta import profsched


class _CountScheduler(profsched._Scheduler): # pylint: disable=protected-access
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
        super().__init__(
            sched.role_names, sched.strat_names, sched.num_role_players)
        utils.check(count > 0, 'count must be positive {:d}', count)
        self._sched = sched
        self._count = count

    async def sample_payoffs(self, profile):
        payoffs = await asyncio.gather(*[
            self._sched.sample_payoffs(profile) for _ in range(self._count)])
        payoff = np.zeros(self.num_strats)
        for i, pay in enumerate(payoffs, 1):
            payoff += (pay - payoff) / i
        return payoff

    def __str__(self):
        return str(self._sched)


def countsched(sched, count):
    """create a count scheduler"""
    return _CountScheduler(sched, count)
