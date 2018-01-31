import mock
import threading

import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis.reduction import deviation_preserving as dpr

from egta import innerloop
from egta import gamesched
from egta import schedgame


games = [
    ([1], [2]),
    ([2], [2]),
    ([2, 1], [1, 2]),
    ([1, 2], [2, 1]),
    ([2, 2], [2, 2]),
    ([3, 2], [2, 3]),
]


def verify_dist_thresh(eqa, thresh=1e-3):
    for i, eqm in enumerate(eqa[:-1], 1):
        assert np.all(thresh ** 2 <= np.sum((eqm - eqa[i:]) ** 2, 1))


@pytest.mark.parametrize('players,strats', games)
def test_innerloop_simple(players, strats):
    sgame = gamegen.samplegame(players, strats)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('players,strats', games)
def test_innerloop_game(players, strats):
    game = gamegen.samplegame(players, strats)
    with gamesched.RsGameScheduler(game) as sched:
        eqas = innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqas)
    eqag = innerloop.inner_loop(game)
    equality = np.isclose(eqas, eqag[:, None]).all(2)
    assert equality.any(1).all() and equality.any(0).all()


@pytest.mark.parametrize('players,strats', games)
def test_innerloop_dpr(players, strats):
    redgame = rsgame.emptygame(players, strats)
    fullgame = rsgame.emptygame(redgame.num_role_players ** 2,
                                redgame.num_role_strats)
    profs = dpr.expand_profiles(fullgame, redgame.all_profiles())
    pays = np.random.random(profs.shape)
    pays[profs == 0] = 0
    sgame = paygame.samplegame_replace(fullgame, profs, [pays[:, None]])
    with gamesched.SampleGameScheduler(sgame) as sched:
        game = schedgame.schedgame(sched, dpr, redgame.num_role_players)
        eqa = innerloop.inner_loop(game)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('players,strats', games)
def test_innerloop_by_role_simple(players, strats):
    sgame = gamegen.samplegame(players, strats)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(
            schedgame.schedgame(sched), devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('when', ['schedule', 'pre-get', 'post-get'])
@pytest.mark.parametrize('count', [1, 5, 10])
@pytest.mark.parametrize('players,strats',
                         [[[2, 3], [3, 2]], [[4, 3], [3, 4]]])
def test_innerloop_failures(players, strats, count, when):
    game = gamegen.game(players, strats)
    with ExceptionScheduler(game, count, when) as sched:
        sgame = schedgame.schedgame(sched)
        with pytest.raises(SchedulerException):
            innerloop.inner_loop(sgame, restricted_game_size=5)


@pytest.mark.parametrize('count', [1, 5])
@pytest.mark.parametrize('players,strats',
                         [[[2, 3], [3, 2]], [[4, 3], [3, 4]]])
def test_threading_failures(players, strats, count):
    game = gamegen.game(players, strats)
    lock = threading.Lock()
    num = [count]
    old_thread = threading.Thread

    def fail_in(*args, **kwargs):
        with lock:
            num[0] -= 1
            if num[0] <= 0:
                return ExceptionThread(*args, **kwargs)
            else:
                return old_thread(*args, **kwargs)

    with gamesched.RsGameScheduler(game) as sched:
        sgame = schedgame.schedgame(sched)
        with mock.patch('threading.Thread', side_effect=fail_in):
            with pytest.raises(ThreadException):
                innerloop.inner_loop(sgame, restricted_game_size=5)


@pytest.mark.parametrize('eq_prob', [x / 10 for x in range(11)])
def test_innerloop_known_eq(eq_prob):
    sgame = paygame.samplegame_copy(gamegen.sym_2p2s_known_eq(eq_prob))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(
            schedgame.schedgame(sched), devs_by_role=True)
    assert eqa.size, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('num', [1, 2])
def test_innerloop_num_eqa(players, strats, num):
    sgame = gamegen.samplegame(players, strats)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(
            schedgame.schedgame(sched),
            num_equilibria=num, devs_by_role=True)
    verify_dist_thresh(eqa)


def test_backups_used():
    """Test that outerloop uses backups

    Since restricted game size is 1, but the only equilibria has support two,
    this must use backups to find an equilibrium."""
    sgame = paygame.samplegame_copy(gamegen.sym_2p2s_known_eq(0.5))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa = innerloop.inner_loop(
            schedgame.schedgame(sched), restricted_game_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


def test_initial_restrictions():
    """Test that outerloop uses backups

    Since restricted game size is 1, but the only equilibria has support two,
    this must use backups to find an equilibrium."""
    game = gamegen.sym_2p2s_known_eq(0.5)
    with gamesched.RsGameScheduler(game) as sched:
        eqa = innerloop.inner_loop(
            schedgame.schedgame(sched), initial_restrictions=[[True, True]],
            restricted_game_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


def test_nash_failure():
    """With regret thresh of zero, nash will fail"""
    game = gamegen.sym_2p2s_known_eq(1 / 3)
    with gamesched.RsGameScheduler(game) as sched:
        eqa = innerloop.inner_loop(schedgame.schedgame(sched), regret_thresh=0)
    assert not eqa.size


class SchedulerException(Exception):
    """Exception to be thrown by ExceptionScheduler"""
    pass


class ExceptionScheduler(gamesched.RsGameScheduler):
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


class ThreadException(Exception):
    """Exception to be thrown by ExceptionScheduler"""
    pass


class ExceptionThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start(self, *args, **kwargs):
        raise ThreadException()
