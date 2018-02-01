import pytest
import numpy as np
from gameanalysis import gamegen
from gameanalysis import paygame

from egta import gamesched
from egta import schedgame


sizes = [
    ([3], [3]),
    ([3, 2], [2, 3]),
    ([2, 2, 2], [2, 2, 2]),
]


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(20))
def test_random_caching(players, strats, _):
    game = gamegen.samplegame(players, strats)
    rest1, rest2 = game.random_restrictions(2)
    mixes = np.random.random((2, game.num_strats))
    mixes *= [rest1, rest2]
    mixes /= np.add.reduceat(mixes, game.role_starts, 1).repeat(
        game.num_role_strats, 1)
    mix1, mix2 = mixes
    with gamesched.SampleGameScheduler(game) as sched:
        sgame = schedgame.schedgame(sched)

        rgame11 = paygame.game_copy(sgame.restrict(rest1))
        rgame21 = paygame.game_copy(sgame.restrict(rest2))
        devs11 = sgame.deviation_payoffs(mix1)
        devs21 = sgame.deviation_payoffs(mix2)

        rgame12 = paygame.game_copy(sgame.restrict(rest1))
        rgame22 = paygame.game_copy(sgame.restrict(rest2))
        assert rgame11 == rgame12
        assert rgame21 == rgame22

        devs12 = sgame.deviation_payoffs(mix1)
        devs22 = sgame.deviation_payoffs(mix2)
        assert np.allclose(devs11, devs12)
        assert np.allclose(devs21, devs22)

        # FIXME
        # assert np.all(rest1 >= rest2) or devg11 != sched.get_data()
        # assert np.all(rest1 <= rest2) or devg21 != sched.get_data()


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(20))
def test_random_complete_dev(players, strats, _):
    game = gamegen.samplegame(players, strats)
    with gamesched.SampleGameScheduler(game) as sched:
        sgame = schedgame.schedgame(sched)
        mix = sgame.random_sparse_mixture()
        supp = mix > 0
        devs, jac = sgame.deviation_payoffs(mix, jacobian=True)
        assert not np.isnan(devs).any()
        assert not np.isnan(jac[supp]).any()
        assert np.isnan(jac[~supp]).all()
        for r in range(sgame.num_roles):
            mask = r == sgame.role_indices
            rdevs = sgame.deviation_payoffs(mix, role_index=r)
            assert np.allclose(rdevs[supp], devs[supp])
            assert np.allclose(rdevs[mask], devs[mask])
            assert supp[~mask].all() or np.isnan(rdevs[~mask]).any()


@pytest.mark.parametrize('players,strats', sizes)
@pytest.mark.parametrize('_', range(20))
def test_random_normalize(players, strats, _):
    game = gamegen.samplegame(players, strats)
    with gamesched.SampleGameScheduler(game) as sched:
        sgame = schedgame.schedgame(sched)

        ngame = sgame.normalize()
        assert np.all(ngame.payoffs() >= -1e-7)
        assert np.all(ngame.payoffs() <= 1 + 1e-7)

        pays = sgame.payoffs().copy()
        pays[sgame.profiles() == 0] = np.nan
        assert np.all(np.nanmin(pays, 0) >= sgame.min_strat_payoffs() - 1e-7)
        assert np.all(np.nanmax(pays, 0) <= sgame.max_strat_payoffs() + 1e-7)
