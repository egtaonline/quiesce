import numpy as np
import pytest
from gameanalysis import gamegen

from egta import bootstrap
from egta import gamesched


SIZES = [
    [[2], [2]],
    [[2, 2], [2, 2]],
    [[3, 2], [2, 3]],
    [[3, 4], [4, 3]],
]


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_mean_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    mix = game.random_mixture()
    with gamesched.RsGameScheduler(game) as sched:
        mean, boot = bootstrap.deviation_payoffs(sched, game, mix, 20)
    assert mean.shape == (game.num_strats,)
    assert boot.shape == (0, game.num_strats)


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_boot_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    mix = game.random_mixture()
    devs = game.deviation_payoffs(mix)
    with gamesched.RsGameScheduler(game) as sched:
        mean, boot = bootstrap.deviation_payoffs(
            sched, game, mix, 20, boots=101, chunk_size=5)
    assert mean.shape == (game.num_strats,)
    assert boot.shape == (101, game.num_strats)
    # These aren't guaranteed to be false, but it's incredibly unlikely
    assert not np.allclose(devs, mean)
    assert not np.allclose(devs, boot)


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_pure_boot_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    with gamesched.RsGameScheduler(game) as sched:
        for mix in game.pure_mixtures():
            devs = game.deviation_payoffs(mix)
            mean, boot = bootstrap.deviation_payoffs(
                sched, game, mix, 20, boots=101)
            assert np.allclose(devs, mean)
            assert np.allclose(devs, boot)
            assert mean.shape == (game.num_strats,)
            assert boot.shape == (101, game.num_strats)
