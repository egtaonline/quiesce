import random

import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame

from egta import outerloop
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


def random_minimal_subgame(game):
    return random.choice(game.pure_subgames())


@pytest.mark.parametrize('size', SIMPLE_SIZES)
def test_innerloop_simple(size):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
def test_innerloop_dpr(size):
    bgame = rsgame.basegame(*size)
    red = reduction.DeviationPreserving(
        bgame.num_strategies, bgame.num_players ** 2, bgame.num_players)
    profs = red.expand_profiles(bgame.all_profiles())
    pays = np.random.random(profs.shape)
    pays[profs == 0] = 0
    sgame = rsgame.samplegame(bgame.num_players ** 2, bgame.num_strategies,
                              profs, [pays[..., None]])
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, red=red)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
def test_innerloop_by_role_simple(size):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
def test_outerloop_simple(size):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    mask = random_minimal_subgame(sgame)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, mask)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
def test_outerloop_by_role_simple(size):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    mask = random_minimal_subgame(sgame)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, mask, devs_by_role=True)
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('eq_prob', [x / 10 for x in range(11)])
def test_outerloop_known_eq(eq_prob):
    sgame = rsgame.samplegame_copy(gamegen.sym_2p2s_known_eq(eq_prob))
    mask = random_minimal_subgame(sgame)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, mask, devs_by_role=True)
    assert eqa.size, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)


@pytest.mark.parametrize('size', SIMPLE_SIZES)
@pytest.mark.parametrize('num', [1, 2])
def test_outerloop_num_eqa(size, num):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game(*size))
    mask = random_minimal_subgame(sgame)
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, outer = outerloop.outer_loop(
            sched, sgame, mask, num_equilibria=num, devs_by_role=True)
    assert eqa.shape[0] >= num or outer.all(), \
        "didn't find enough equilibrium and stopped"
    verify_dist_thresh(eqa)


def test_backups_used():
    """Test that outerloop uses backups

    Since subgame size is 1, but the only equilibria has support two, this must
    use backups to find an equilibrium."""
    sgame = rsgame.samplegame_copy(gamegen.sym_2p2s_known_eq(0.5))
    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame, subgame_size=1)
    assert eqa.size, "didn't find equilibrium"
    expected = [0.5, 0.5]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(-1).any()
    verify_dist_thresh(eqa)
