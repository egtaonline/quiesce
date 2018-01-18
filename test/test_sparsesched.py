import pytest
import numpy as np
from gameanalysis import gamegen
from gameanalysis import rsgame
from gameanalysis import subgame

from egta import gamesched
from egta import sparsesched


sizes = [
    ([3], [3]),
    ([3, 2], [2, 3]),
    ([2, 2, 2], [2, 2, 2]),
]


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(5))
def test_random_caching(players, strats, _):
    game = gamegen.add_noise(
        gamegen.add_profiles(rsgame.emptygame(players, strats)))
    sub1, sub2 = game.random_subgames(2)
    with gamesched.SampleGameScheduler(game) as prof_sched:
        sched = sparsesched.SparseScheduler(prof_sched)

        subg11 = sched.get_subgame(sub1, 1)
        subg21 = sched.get_subgame(sub2, 1)
        devg11 = sched.get_deviations(sub1, 1)
        devg21 = sched.get_deviations(sub2, 1)

        subg12 = sched.get_subgame(sub1, 1)
        subg22 = sched.get_subgame(sub2, 1)
        assert subg11 == subg12
        assert subg21 == subg22

        devg12 = sched.get_deviations(sub1, 1)
        devg22 = sched.get_deviations(sub2, 1)
        assert devg11 == devg12
        assert devg21 == devg22

        assert np.all(sub1 >= sub2) or devg11 != sched.get_data()
        assert np.all(sub1 <= sub2) or devg21 != sched.get_data()


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(5))
def test_random_complete_dev(players, strats, _):
    egame = rsgame.emptygame(players, strats)
    game = gamegen.add_noise(gamegen.add_profiles(egame))
    sub = game.random_subgame()
    with gamesched.SampleGameScheduler(game) as prof_sched:
        sched = sparsesched.SparseScheduler(prof_sched)
        mix = subgame.translate(egame.subgame(sub).random_mixture(), sub)
        devs = sched.get_deviations(sub, 1)
        assert not np.isnan(devs.deviation_payoffs(mix)).any()
