"""Module for test utilities"""
from egta import profsched


GAMES = [
    ([1], [2]),
    ([2], [2]),
    ([2, 1], [1, 2]),
    ([1, 2], [2, 1]),
    ([2, 2], [2, 2]),
    ([3, 2], [2, 3]),
    ([1, 1, 1], [2, 2, 2]),
]


class SchedulerException(Exception):
    """Exception to be thrown by ExceptionScheduler"""
    pass


class ExceptionScheduler(profsched._Scheduler): # pylint: disable=protected-access
    """Scheduler that allows triggering exeptions on command"""

    def __init__(self, base, error_after, call_type):
        super().__init__(
            base.role_names, base.strat_names, base.num_role_players)
        self._base = base
        self._calls = 0
        self._error_after = error_after
        self._call_type = call_type

    async def sample_payoffs(self, profile):
        self._calls += 1
        if self._error_after <= self._calls and self._call_type == 'pre':
            raise SchedulerException
        pay = await self._base.sample_payoffs(profile)
        if self._error_after <= self._calls and self._call_type == 'post':
            raise SchedulerException
        return pay
