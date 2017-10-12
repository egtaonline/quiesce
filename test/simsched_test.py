import json
import pytest
import time

import numpy as np
from gameanalysis import gameio
from gameanalysis import reduction

from egta import simsched
from egta import innerloop


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
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)


def test_delayed_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    prof = game.random_profiles()
    cmd = ['bash', '-c', 'sleep 1 && false']

    with pytest.raises(RuntimeError):
        with simsched.SimulationScheduler(serial, conf, cmd) as sched:
            with pytest.raises(RuntimeError):
                sched.schedule(prof).get()


def test_early_exit():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)
    cmd = ['bash', '-c', 'while read line; do : ; done']

    finished_scheduling = False
    with pytest.raises(RuntimeError):
        with simsched.SimulationScheduler(serial, conf, cmd) as sched:
            # Schedule promises but just exit
            for p in profs:
                sched.schedule(p)
            finished_scheduling = True
    assert finished_scheduling, \
        "didn't finish scheduling"


def test_read_delay_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    cmd = ['bash', '-c', 'read line && false']

    with pytest.raises(RuntimeError):
        with simsched.SimulationScheduler(serial, conf, cmd) as sched:
            sched.schedule(game.random_profiles()).get()


def test_read_delay_schedule_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    cmd = ['bash', '-c', 'read line && false']

    got_here = False
    with pytest.raises(RuntimeError):
        with simsched.SimulationScheduler(serial, conf, cmd) as sched:
            sched.schedule(game.random_profiles())
            with pytest.raises(RuntimeError):
                got_here = True
                sched.schedule(game.random_profiles())
    assert got_here, \
        "didn't get to second schedule"


def test_ignore_terminate_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    cmd = ['bash', '-c', 'trap "" SIGTERM && sleep 60']
    with simsched.SimulationScheduler(serial, conf, cmd):
        # Wait for term to be captured
        time.sleep(1)


@pytest.mark.long
def test_innerloop_simple():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game, serial = gameio.read_basegame(jgame)
    cmd = ['python3', 'cdasim/sim.py', '--single', '1']
    red = reduction.DeviationPreserving(
        game.num_strategies, game.num_players, np.repeat(2, game.num_roles))

    with simsched.SimulationScheduler(serial, conf, cmd) as sched:
        innerloop.inner_loop(sched, game, red=red)
