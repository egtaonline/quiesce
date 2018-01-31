import numpy as np
import pytest
from gameanalysis import gamegen

from egta import gamesched
from egta import normsched


@pytest.mark.parametrize('players,strats', [
    ([2], [2]), ([1, 2], [2, 1]), ([2, 1], [1, 2]), ([2, 1, 2], [1, 2, 1]),
    ([1, 2, 1], [2, 1, 2]),
])
def test_basic_profile(players, strats):
    game = gamegen.game(players, strats)
    with normsched.NormScheduler(gamesched.RsGameScheduler(game)) as sched:
        ngame = sched.game()
        assert np.all(ngame.num_role_strats > 1)
        pay = sched.schedule(ngame.random_profile()).get()
        assert pay.size == ngame.num_strats
