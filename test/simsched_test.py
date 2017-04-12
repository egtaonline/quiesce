import json

import numpy as np
from gameanalysis import gameio
from gameanalysis import reduction

from egta import simsched
from egta import outerloop


def test_basic_profile():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)
    cmd = ['python3', 'cdasim/sim.py', '--single', '1']

    with simsched.SimulationScheduler(serial, conf, cmd) as sched:
        proms = [sched.schedule(p) for p in profs]
        pays = np.concatenate([p.get()[None] for p in proms])
    assert np.allclose(pays[profs == 0], 0)


def test_innerloop_simple():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    cmd = ['python3', 'cdasim/sim.py', '--single', '1']
    red = reduction.DeviationPreserving(
        game.num_strategies, game.num_players, np.repeat(2, game.num_roles))

    with simsched.SimulationScheduler(serial, conf, cmd) as sched:
        eqa, _ = outerloop.outer_loop(sched, game, red=red)
