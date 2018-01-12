import numpy as np
import numpy.random as rand
from gameanalysis import gamegen
from gameanalysis import agggen

from egta import gamesched


def test_basic_profile():
    game = gamegen.role_symmetric_game([4, 3], [3, 4])
    profs = game.random_profiles(20)

    with gamesched.RsGameScheduler(game) as sched:
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


def test_duplicate_profile_sample():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 1)
    profs = sgame.random_profiles(20)

    with gamesched.SampleGameScheduler(sgame) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays1 = np.concatenate([p.get()[None] for p in proms])
        proms = [sched.schedule(p) for p in profs]
        pays2 = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays1[profs == 0], 0)
    assert np.allclose(pays2[profs == 0], 0)
    assert np.allclose(pays1, pays2)


def test_basic_profile_aggfn():
    agame = agggen.normal_aggfn([4, 3], [3, 4], 5)
    profs = agame.random_profiles(20)

    with gamesched.RsGameScheduler(agame) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_noise_profile():
    sgame = gamegen.add_noise(
        gamegen.role_symmetric_game([4, 3], [3, 4]), 1, 3)
    profs = sgame.random_profiles(20)

    sched = gamesched.SampleGameScheduler(
        sgame, lambda w: rand.normal(0, w, sgame.num_strats),
        lambda: (rand.random(),))
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_duplicate_prof():
    game = gamegen.role_symmetric_game([4, 3], [3, 4])
    profs = game.random_profiles(20)

    with gamesched.RsGameScheduler(game) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)

        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)
