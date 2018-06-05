"""Test inner loop"""
import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import deviation_preserving as dpr

from egta import asyncgame
from egta import innerloop
from egta import gamesched
from egta import schedgame
from test import utils as tu # pylint: disable=wrong-import-order


def verify_dist_thresh(eqa, thresh=0.1):
    """Verify that all equilibria are different by at least thresh"""
    for i, eqm in enumerate(eqa[:-1], 1):
        assert np.all(thresh ** 2 <= np.sum((eqm - eqa[i:]) ** 2, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', tu.GAMES)
@pytest.mark.parametrize('_', range(1))
async def test_innerloop_simple(players, strats, _):
    """Test that inner loop works for a simple setup"""
    sgame = gamegen.samplegame(players, strats)
    sched = gamesched.samplegamesched(sgame)
    eqa = await innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', tu.GAMES)
async def test_innerloop_game(players, strats):
    """Test that inner loop works for a game"""
    game = gamegen.samplegame(players, strats)
    sched = gamesched.gamesched(game)
    eqas = await innerloop.inner_loop(schedgame.schedgame(sched))
    verify_dist_thresh(eqas)
    eqag = await innerloop.inner_loop(asyncgame.wrap(game))
    assert utils.allclose_perm(eqas, eqag, atol=1e-3, rtol=1e3)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', tu.GAMES)
async def test_innerloop_dpr(players, strats):
    """Test that inner loop handles dpr"""
    redgame = rsgame.empty(players, strats)
    fullgame = rsgame.empty(
        redgame.num_role_players ** 2, redgame.num_role_strats)
    profs = dpr.expand_profiles(fullgame, redgame.all_profiles())
    pays = np.random.random(profs.shape)
    pays[profs == 0] = 0
    sgame = paygame.samplegame_replace(fullgame, profs, [pays[:, None]])
    sched = gamesched.samplegamesched(sgame)
    game = schedgame.schedgame(sched, dpr, redgame.num_role_players)
    eqa = await innerloop.inner_loop(game)
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', tu.GAMES)
async def test_innerloop_by_role_simple(players, strats):
    """Test that inner loop works by role"""
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
    """Test that inner loop handles exceptions during scheduling"""
    game = gamegen.game(players, strats)
    sched = ExceptionScheduler(game, count, when)
    sgame = schedgame.schedgame(sched)
    with pytest.raises(SchedulerException):
        await innerloop.inner_loop(sgame, restricted_game_size=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('eq_prob', [x / 10 for x in range(11)])
async def test_innerloop_known_eq(eq_prob):
    """Test that inner loop finds known equilibria"""
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    sched = gamesched.gamesched(game)
    eqa = await innerloop.inner_loop(
        schedgame.schedgame(sched), devs_by_role=True)
    assert eqa.size, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', tu.GAMES)
@pytest.mark.parametrize('num', [1, 2])
async def test_innerloop_num_eqa(players, strats, num):
    """Test that inner loop returns alternate number of equilibria"""
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
@pytest.mark.parametrize('players,strats', tu.GAMES)
@pytest.mark.parametrize('_', range(5))
async def test_at_least_one(players, strats, _):
    """inner loop should always find one equilibrium with at_least one"""
    game = gamegen.game(players, strats)
    eqa = await innerloop.inner_loop(asyncgame.wrap(game), at_least_one=True)
    assert eqa.size


class SchedulerException(Exception):
    """Exception to be thrown by ExceptionScheduler"""
    pass


class ExceptionScheduler(gamesched._RsGameScheduler): # pylint: disable=protected-access
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
