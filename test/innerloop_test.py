import threading

import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame

from egta import innerloop
from egta import gamesched


SIMPLE_SIZES = [
    [[1], [1]],
    [[2], [2]],
    [[2, 2], [2, 2]],
    [[3, 2], [2, 3]],
    [[3, 4], [4, 3]],
]


def verify_dist_thresh(eqa, thresh=1e-3):
    for i, eqm in enumerate(eqa[:-1], 1):
        assert np.all(thresh ** 2 <= np.sum((eqm - eqa[i:]) ** 2, 1))


@pytest.mark.parametrize('players,strats', SIMPLE_SIZES)
def test_innerloop_simple(players, strats):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(
        players, strats))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(sched, sgame)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('players,strats', SIMPLE_SIZES)
def test_innerloop_dpr(players, strats):
    bgame = rsgame.basegame(players, strats)
    red = reduction.DeviationPreserving(
        bgame.num_strategies, bgame.num_players ** 2, bgame.num_players)
    profs = red.expand_profiles(bgame.all_profiles())
    pays = np.random.random(profs.shape)
    pays[profs == 0] = 0
    sgame = rsgame.samplegame(bgame.num_players ** 2, bgame.num_strategies,
                              profs, [pays[..., None]])
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(sched, sgame, red=red)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('players,strats', SIMPLE_SIZES)
def test_innerloop_by_role_simple(players, strats):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(
        players, strats))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(sched, sgame, devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('when', ['schedule', 'pre-get', 'post-get'])
@pytest.mark.parametrize('count', [1, 5, 10, 20])
@pytest.mark.parametrize('players,strats',
                         [[[2, 3], [3, 2]], [[4, 3], [3, 4]]])
def test_innerloop_failures(players, strats, count, when):
    game = gamegen.role_symmetric_game(players, strats)
    with ExceptionScheduler(game, count, when) as sched:
        with pytest.raises(SchedulerException):
            innerloop.inner_loop(sched, game, subgame_size=5)


@pytest.mark.parametrize('eq_prob', [x / 10 for x in range(11)])
def test_innerloop_known_eq(eq_prob):
    sgame = rsgame.samplegame_copy(gamegen.sym_2p2s_known_eq(eq_prob))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(sched, sgame, devs_by_role=True)
    assert eqa.size, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
@pytest.mark.parametrize('num', [1, 2])
def test_innerloop_num_eqa(size, num):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(
            sched, sgame, num_equilibria=num, devs_by_role=True)
    verify_dist_thresh(eqa)


def test_backups_used():
    """Test that outerloop uses backups

    Since subgame size is 1, but the only equilibria has support two, this must
    use backups to find an equilibrium."""
    sgame = rsgame.samplegame_copy(gamegen.sym_2p2s_known_eq(0.5))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(sched, sgame, subgame_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


class SchedulerException(Exception):
    """Exception to be thrown by ExceptionScheduler"""
    pass


class ExceptionScheduler(gamesched.GameScheduler):
    """Scheduler that allows triggering exeptions on command"""

    def __init__(self, game, error_after, call_type):
        super().__init__(game)
        self._calls = 0
        self._error_after = error_after
        self._call_type = call_type
        self._lock = threading.Lock()

    def _inc_and_check(self, name):
        if self._call_type == name:
            with self._lock:
                self._calls += 1
                if self._error_after <= self._calls:
                    raise SchedulerException

    def schedule(self, profile):
        self._inc_and_check('schedule')
        return ExceptionPromise(self, super().schedule(profile))


class ExceptionPromise(object):
    """Promise for ExceptionScheduler"""

    def __init__(self, sched, prom):
        self._sched = sched
        self._prom = prom

    def get(self):
        self._sched._inc_and_check('pre-get')
        res = self._prom.get()
        self._sched._inc_and_check('post-get')
        return res
