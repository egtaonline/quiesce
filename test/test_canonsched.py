import numpy as np
import pytest
from gameanalysis import gamegen

from egta import canonsched
from egta import gamesched


@pytest.mark.asyncio
@pytest.mark.parametrize('players,strats', [
    ([2], [2]), ([1, 2], [2, 1]), ([2, 1], [1, 2]), ([2, 1, 2], [1, 2, 1]),
    ([1, 2, 1], [2, 1, 2]),
])
async def test_basic_profile(players, strats):
    game = gamegen.game(players, strats)
    sched = canonsched.canon(gamesched.gamesched(game))
    assert np.all(sched.num_role_strats > 1)
    pay = await sched.sample_payoffs(sched.random_profile())
    assert pay.size == sched.num_strats
