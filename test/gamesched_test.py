import numpy as np
import numpy.random as rand
from gameanalysis import gamegen
from gameanalysis import agggen

from egta import gamesched
from egta import outerloop


def test_basic_profile():
    game = gamegen.role_symmetric_game([4, 3], [3, 4])
    profs = game.random_profiles(20)

    with gamesched.GameScheduler(game) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_basic_profile_sample():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 3)
    profs = sgame.random_profiles(20)

    with gamesched.SampleGameScheduler(sgame) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_basic_profile_aggfn():
    agame = agggen.random_aggfn([4, 3], [3, 4], 5)
    profs = agame.random_profiles(20)

    with gamesched.AggfnScheduler(agame) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_noise_profile():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 3)
    profs = sgame.random_profiles(20)

    sched = gamesched.SampleGameScheduler(
        sgame, lambda w: rand.normal(0, w, sgame.num_role_strats),
        lambda: (rand.random(),))
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_innerloop_simple():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 3)

    with gamesched.SampleGameScheduler(sgame) as sched:
        eqa, _ = outerloop.outer_loop(sched, sgame)
