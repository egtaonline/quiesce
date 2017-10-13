import json
import pytest
import random

import numpy as np
from gameanalysis import gameio

from egta import countsched
from egta import eosched


sleep_time = 60
sim_memory = 2048
sim_time = 60


# FIXME Ideally we want a mock EgtaOnlineApi

@pytest.mark.egta
def test_basic_profile():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    # Guarantees that new profs will be scheduled
    conf['nonce'] = random.randrange(2**31)
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)

    # Schedule all new profiles and verify it works
    sched = eosched.EgtaOnlineScheduler(
        107, game, serial, 1, conf, sleep_time, 25, sim_memory, sim_time)
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)
    assert sched._num_running_profiles == 0

    # Schedule old profiles and verify it still works
    sched = eosched.EgtaOnlineScheduler(
        107, game, serial, 1, conf, sleep_time, 25, sim_memory, sim_time)
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)
    assert sched._num_running_profiles == 0

    # Schedule two at a time, in two batches
    base_sched = eosched.EgtaOnlineScheduler(
        107, game, serial, 2, conf, sleep_time, 15, sim_memory, sim_time)
    sched = countsched.CountScheduler(base_sched, 2)
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)
    assert base_sched._num_running_profiles == 0

    # Try again now that everything should be scheduled
    base_sched = eosched.EgtaOnlineScheduler(
        107, game, serial, 2, conf, sleep_time, 15, sim_memory, sim_time)
    sched = countsched.CountScheduler(base_sched, 2)
    with sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)
    assert base_sched._num_running_profiles == 0
