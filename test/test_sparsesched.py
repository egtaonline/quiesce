import pytest
import numpy as np
from gameanalysis import gamegen
from gameanalysis import restrict
from gameanalysis import rsgame

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
    game = gamegen.samplegame(players, strats)
    rest1, rest2 = game.random_restrictions(2)
    with gamesched.SampleGameScheduler(game) as prof_sched:
        sched = sparsesched.SparseScheduler(prof_sched)

        rgame11 = sched.get_restricted_game(rest1, 1)
        rgame21 = sched.get_restricted_game(rest2, 1)
        devg11 = sched.get_deviations(rest1, 1)
        devg21 = sched.get_deviations(rest2, 1)

        rgame12 = sched.get_restricted_game(rest1, 1)
        rgame22 = sched.get_restricted_game(rest2, 1)
        assert rgame11 == rgame12
        assert rgame21 == rgame22

        devg12 = sched.get_deviations(rest1, 1)
        devg22 = sched.get_deviations(rest2, 1)
        assert devg11 == devg12
        assert devg21 == devg22

        assert np.all(rest1 >= rest2) or devg11 != sched.get_data()
        assert np.all(rest1 <= rest2) or devg21 != sched.get_data()


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(5))
def test_random_complete_dev(players, strats, _):
    egame = rsgame.emptygame(players, strats)
    game = gamegen.gen_noise(gamegen.gen_profiles(egame))
    rest = game.random_restriction()
    with gamesched.SampleGameScheduler(game) as prof_sched:
        sched = sparsesched.SparseScheduler(prof_sched)
        mix = restrict.translate(egame.restrict(rest).random_mixture(), rest)
        devs = sched.get_deviations(rest, 1)
        assert not np.isnan(devs.deviation_payoffs(mix)).any()
