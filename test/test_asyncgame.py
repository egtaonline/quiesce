import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import restrict
from gameanalysis import rsgame

from egta import asyncgame


@pytest.mark.asyncio
async def test_basic_asyncgame():
    game = gamegen.game([4, 3], [3, 4])
    agame = asyncgame.wrap(game)
    rest = agame.random_restriction()
    rgame = await agame.get_restricted_game(rest)
    assert rgame.is_complete()
    assert (rsgame.emptygame_copy(rgame) ==
            rsgame.emptygame_copy(game.restrict(rest)))

    dgame = await agame.get_deviation_game(rest)
    mix = restrict.translate(rgame.random_mixture(), rest)
    assert not np.isnan(dgame.deviation_payoffs(mix)).any()

    prof = agame.random_profile()

    with pytest.raises(ValueError):
        agame.num_complete_profiles
    with pytest.raises(ValueError):
        agame.num_profiles
    with pytest.raises(ValueError):
        agame.profiles()
    with pytest.raises(ValueError):
        agame.payoffs()
    with pytest.raises(ValueError):
        agame.min_strat_payoffs()
    with pytest.raises(ValueError):
        agame.max_strat_payoffs()
    with pytest.raises(ValueError):
        agame.get_payoffs(prof)
    with pytest.raises(ValueError):
        agame.deviation_payoffs(mix)
    with pytest.raises(ValueError):
        agame.normalize()
    with pytest.raises(ValueError):
        agame.restrict(rest)
    with pytest.raises(ValueError):
        prof in agame
