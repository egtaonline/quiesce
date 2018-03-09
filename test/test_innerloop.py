import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis.reduction import deviation_preserving as dpr

from egta import asyncgame
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
    ([1, 1, 1], [2, 2, 2]),
]


def verify_dist_thresh(eqa, thresh=0.1):
    for i, eqm in enumerate(eqa[:-1], 1):
        assert np.all(thresh ** 2 <= np.sum((eqm - eqa[i:]) ** 2, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('_', range(1))
async def test_innerloop_simple(players, strats, _):
    sgame = gamegen.samplegame(players, strats)
    sched = gamesched.samplegamesched(sgame)
    eqa = await innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
async def test_innerloop_game(players, strats):
    game = gamegen.samplegame(players, strats)
    sched = gamesched.gamesched(game)
    eqas = await innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqas)
    eqag = await innerloop.inner_loop(asyncgame.wrap(game))
    equality = np.isclose(eqas, eqag[:, None], atol=1e-3).all(2)
    assert equality.any(1).all() and equality.any(0).all()


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
async def test_innerloop_dpr(players, strats):
    redgame = rsgame.emptygame(players, strats)
    fullgame = rsgame.emptygame(redgame.num_role_players ** 2,
                                redgame.num_role_strats)
    profs = dpr.expand_profiles(fullgame, redgame.all_profiles())
    pays = np.random.random(profs.shape)
    pays[profs == 0] = 0
    sgame = paygame.samplegame_replace(fullgame, profs, [pays[:, None]])
    sched = gamesched.samplegamesched(sgame)
    game = schedgame.schedgame(sched, dpr, redgame.num_role_players)
    eqa = await innerloop.inner_loop(game)
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
async def test_innerloop_by_role_simple(players, strats):
    sgame = gamegen.samplegame(players, strats)
    sched = gamesched.samplegamesched(sgame)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('when', ['pre', 'post'])
@pytest.mark.parametrize('count', [1, 5, 10])
@pytest.mark.parametrize('players,strats',
                         [[[2, 3], [3, 2]], [[4, 3], [3, 4]]])
async def test_innerloop_failures(players, strats, count, when):
    game = gamegen.game(players, strats)
    sched = ExceptionScheduler(game, count, when)
    sgame = schedgame.schedgame(sched)
    with pytest.raises(SchedulerException):
        await innerloop.inner_loop(sgame, restricted_game_size=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('eq_prob', [x / 10 for x in range(11)])
async def test_innerloop_known_eq(eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    sched = gamesched.gamesched(game)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), devs_by_role=True)
    assert eqa.size, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('num', [1, 2])
async def test_innerloop_num_eqa(players, strats, num):
    sgame = gamegen.samplegame(players, strats)
    sched = gamesched.samplegamesched(sgame)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched),
        num_equilibria=num, devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
async def test_backups_used():
    """Test that outerloop uses backups

    Since restricted game size is 1, but the only equilibria has support two,
    this must use backups to find an equilibrium."""
    sgame = gamegen.sym_2p2s_known_eq(0.5)
    sched = gamesched.gamesched(sgame)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), restricted_game_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
async def test_initial_restrictions():
    """Test that outerloop uses backups

    Since restricted game size is 1, but the only equilibria has support two,
    this must use backups to find an equilibrium."""
    game = gamegen.sym_2p2s_known_eq(0.5)
    sched = gamesched.gamesched(game)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), initial_restrictions=[[True, True]],
        restricted_game_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
async def test_nash_failure():
    """With regret thresh of zero, nash will fail"""
    game = gamegen.sym_2p2s_known_eq(1 / 3)
    sched = gamesched.gamesched(game)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), regret_thresh=0)
    assert not eqa.size


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('_', range(5))
async def test_at_least_one(players, strats, _):
    """inner loop should always find one equilibrium with at_least one"""
    game = gamegen.game(players, strats)
    eqa = await innerloop.inner_loop(asyncgame.wrap(game), at_least_one=True)
    assert eqa.size


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

    async def sample_payoffs(self, profile):
        self._calls += 1
        if self._error_after <= self._calls and self._call_type == 'pre':
            raise SchedulerException
        pay = await super().sample_payoffs(profile)
        if self._error_after <= self._calls and self._call_type == 'post':
            raise SchedulerException
        return pay
