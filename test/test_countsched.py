import numpy as np
from gameanalysis import gamegen
from gameanalysis import utils

from egta import countsched
from egta import gamesched
from egta import savesched


# TODO Use savesched to test that is is scheduling 10
def test_basic_profile():
    sgame = gamegen.samplegame([4, 3], [3, 4])
    profs = utils.unique_axis(sgame.random_profiles(20))

    save = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    with countsched.CountScheduler(save, 10) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.stack([p.get() for p in proms])
        pays2 = np.stack([p.get() for p in proms])
    assert np.allclose(pays[profs == 0], 0)
    assert np.allclose(pays, pays2)

    savegame = save.get_samplegame()
    assert list(savegame.num_samples) == [10]
