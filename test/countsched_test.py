import numpy as np
from gameanalysis import gamegen

from egta import gamesched
from egta import countsched


def test_basic_profile():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 3)
    profs = sgame.random_profiles(20)

    sched = countsched.CountScheduler(gamesched.SampleGameScheduler(sgame), 10)
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)
