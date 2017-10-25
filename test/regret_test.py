import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import regret as greg

from egta import gamesched
from egta import regret


SIZES = [
    [[2], [2]],
    [[2, 2], [2, 2]],
    [[3, 2], [2, 3]],
    [[3, 4], [4, 3]],
]


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_mean_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    mix = game.random_mixtures()
    with gamesched.RsGameScheduler(game) as sched:
        mean, boot = regret.boot(sched, game, mix, 20)
    assert mean.shape == (game.num_strats,)
    assert boot.shape == (0, game.num_strats)


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_boot_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    mix = game.random_mixtures()
    gains = greg.mixture_deviation_gains(game, mix)
    with gamesched.RsGameScheduler(game) as sched:
        mean, boot = regret.boot(sched, game, mix, 20, boots=101, chunk_size=5)
    assert mean.shape == (game.num_strats,)
    assert boot.shape == (101, game.num_strats)
    # These aren't guaranteed to be false, but it's incredibly unlikely
    assert not np.allclose(gains, mean)
    assert not np.allclose(gains, boot)


@pytest.mark.parametrize('players,strats', SIZES)
def test_random_pure_boot_reg(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    with gamesched.RsGameScheduler(game) as sched:
        for mix in game.pure_mixtures():
            gains = greg.mixture_deviation_gains(game, mix)
            mean, boot = regret.boot(sched, game, mix, 20, boots=101)
            assert np.allclose(gains, mean)
            assert np.allclose(gains, boot)
            assert mean.shape == (game.num_strats,)
            assert boot.shape == (101, game.num_strats)
