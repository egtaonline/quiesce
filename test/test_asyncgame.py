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

    dup = asyncgame.wrap(game)
    assert hash(dup) == hash(agame)
    assert dup == agame


@pytest.mark.asyncio
async def test_mix_asyncgame():
    game0 = gamegen.game([4, 3], [3, 4])
    game1 = gamegen.game([4, 3], [3, 4])
    agame = asyncgame.mix(asyncgame.wrap(game0), asyncgame.wrap(game1), 0.4)
    assert str(agame) == '{} - 0.4 - {}'.format(repr(game0), repr(game1))

    rest = agame.random_restriction()
    rgame = await agame.get_restricted_game(rest)
    assert rgame.is_complete()
    assert (rsgame.emptygame_copy(rgame) ==
            rsgame.emptygame_copy(game0.restrict(rest)))

    dgame = await agame.get_deviation_game(rest)
    mix = restrict.translate(rgame.random_mixture(), rest)
    assert not np.isnan(dgame.deviation_payoffs(mix)).any()

    dup = asyncgame.mix(asyncgame.wrap(game0), asyncgame.wrap(game1), 0.4)
    assert hash(dup) == hash(agame)
    assert dup == agame
