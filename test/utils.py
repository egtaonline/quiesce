"""Module for test utilities"""
import pytest
import timeout_decorator
from gameanalysis import rsgame

from egta import profsched


def games():
    """Test games"""
    yield rsgame.empty(1, 2)
    yield rsgame.empty(2, 2)
    yield rsgame.empty([2, 1], [1, 2])
    yield rsgame.empty([1, 2], [2, 1])
    yield rsgame.empty([2, 2], [2, 2])
    yield rsgame.empty([3, 2], [2, 3])
    yield rsgame.empty([1, 1, 1], 2)


def timeout(seconds):
    """Timeout test without error"""

    def decorator(func):
        """Decorator of function"""
        for decorator in [
            timeout_decorator.timeout(seconds),
            pytest.mark.xfail(raises=timeout_decorator.timeout_decorator.TimeoutError),
        ]:
            func = decorator(func)
        return func

    return decorator


class SchedulerException(Exception):
    """Exception to be thrown by ExceptionScheduler"""

    pass


class ExceptionScheduler(profsched._Scheduler):  # pylint: disable=protected-access
    """Scheduler that allows triggering exeptions on command"""

    def __init__(self, base, error_after, call_type):
        super().__init__(base.role_names, base.strat_names, base.num_role_players)
        self._base = base
        self._calls = 0
        self._error_after = error_after
        self._call_type = call_type

    async def sample_payoffs(self, profile):
        self._calls += 1
        if self._error_after <= self._calls and self._call_type == "pre":
            raise SchedulerException
        pay = await self._base.sample_payoffs(profile)
        if self._error_after <= self._calls and self._call_type == "post":
            raise SchedulerException
        return pay
