import json
import pytest
import random
import time

import numpy as np
from egtaonline import mockapi
from gameanalysis import gameio

from egta import countsched
from egta import eosched


def test_basic_profile():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)

    with mockapi.EgtaOnlineApi() as api:
        # Setup api
        sim = api.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10)
        sim.add_dict({role: strats for role, strats
                      in zip(serial.role_names, serial.strat_names)})

        # Schedule all new profiles and verify it works
        # This first time should have to wait to schedule more
        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 10, 0, 0)
        with sched:
            proms = [sched.schedule(p) for p in profs]
            pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)
        assert sched._num_running_profiles == 0

        # Schedule old profiles and verify it still works
        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        with sched:
            proms = [sched.schedule(p) for p in profs]
            pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)
        assert sched._num_running_profiles == 0

        # Schedule two at a time, in two batches
        base_sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        sched = countsched.CountScheduler(base_sched, 2)
        with sched:
            proms = [sched.schedule(p) for p in profs]
            pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)
        assert base_sched._num_running_profiles == 0

        # Try again now that everything should be scheduled
        base_sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        sched = countsched.CountScheduler(base_sched, 2)
        with sched:
            proms = [sched.schedule(p) for p in profs]
            pays = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays[profs == 0], 0)
        assert base_sched._num_running_profiles == 0


def test_extra_samples():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)

    with mockapi.EgtaOnlineApi() as api:
        # Setup api
        sim = api.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10)
        sim.add_dict({role: strats for role, strats
                      in zip(serial.role_names, serial.strat_names)})

        # Schedule all new profiles and verify it works
        # This first time should have to wait to schedule more
        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        with sched:
            proms = [sched.schedule(p) for p in profs]
            pays1 = np.concatenate([p.get()[None] for p in proms])
            proms = [sched.schedule(p) for p in profs]
            pays2 = np.concatenate([p.get()[None] for p in proms])
        assert np.allclose(pays1[profs == 0], 0)
        assert np.allclose(pays2[profs == 0], 0)
        assert sched._num_running_profiles == 0


def test_exception_in_sechedule():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)

    with mockapi.ExceptionEgtaOnlineApi(TimeoutError, 60) as api:
        # Setup api
        sim = api.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10)
        sim.add_dict({role: strats for role, strats
                      in zip(serial.role_names, serial.strat_names)})

        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        opened = False
        with pytest.raises(TimeoutError):
            with sched:
                opened = True
                with pytest.raises(TimeoutError):
                    for p in profs:
                        sched.schedule(p)
        assert opened


def test_exception_in_get():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    game, serial = gameio.read_basegame(jgame)
    profs = game.random_profiles(20)

    with mockapi.ExceptionEgtaOnlineApi(TimeoutError, 110) as api:
        # Setup api
        sim = api.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10)
        sim.add_dict({role: strats for role, strats
                      in zip(serial.role_names, serial.strat_names)})

        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 1, 25, 0, 0)
        scheduled = False
        with pytest.raises(TimeoutError):
            with sched:
                proms = [sched.schedule(p) for p in profs]
                scheduled = True
                with pytest.raises(TimeoutError):
                    for p in proms:
                        p.get()
        assert scheduled


def test_exception_pre_schedule():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    game, serial = gameio.read_basegame(jgame)
    prof = game.random_profiles()

    with mockapi.ExceptionEgtaOnlineApi(TimeoutError, 60) as api:
        # Setup api
        sim = api.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10)
        sim.add_dict({role: strats for role, strats
                      in zip(serial.role_names, serial.strat_names)})

        sched = eosched.EgtaOnlineScheduler(
            api, sim['id'], game, serial, 1, {}, 0.1, 25, 0, 0)
        slept = False
        with pytest.raises(TimeoutError):
            with sched:
                # so that enough calls to get_requirements are made
                time.sleep(1)
                slept = True
                with pytest.raises(TimeoutError):
                    sched.schedule(prof)
        assert slept
