import asyncio
import random

import numpy as np
import pytest
from egtaonline import api
from egtaonline import mockserver
from gameanalysis import rsgame

from egta import countsched
from egta import eosched


# TODO Switch to fixtures etc, especially with context managers


@pytest.fixture
def egta_info():
    game = rsgame.emptygame([4, 4], [11, 11])
    with mockserver.Server() as server, api.EgtaOnlineApi() as egta:
        sim = egta.get_simulator(server.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10))
        sim.add_dict({role: strats for role, strats
                      in zip(game.role_names, game.strat_names)})
        yield server, egta, game, sim


@pytest.mark.asyncio
async def test_basic_profile(egta_info):
    _, egta, game, sim = egta_info
    profs = game.random_profiles(20)

    # Schedule all new profiles and verify it works
    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 10, 0, 0) as sched:
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Schedule old profiles and verify it still works
    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 10, 0, 0) as sched:
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Schedule two at a time, in two batches
    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 25, 0, 0) as base_sched:
        sched = countsched.CountScheduler(base_sched, 2)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Try again now that everything should be scheduled
    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 25, 0, 0) as base_sched:
        sched = countsched.CountScheduler(base_sched, 2)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)


@pytest.mark.asyncio
async def test_exception_in_create(egta_info):
    server, egta, game, sim = egta_info
    server.throw_exception(TimeoutError)
    with pytest.raises(TimeoutError):
        async with eosched.EgtaOnlineScheduler(
                game, egta, sim['id'], 1, {}, 0.1, 25, 0, 0):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_exception_in_get(egta_info):
    server, egta, game, sim = egta_info
    profs = game.random_profiles(20)

    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 10, 0, 0) as sched:
        futures = asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
        await asyncio.sleep(0.1)
        server.throw_exception(TimeoutError)
        await asyncio.sleep(0.1)
        with pytest.raises(TimeoutError):
            await futures


@pytest.mark.asyncio
async def test_exception_in_schedule(egta_info):
    server, egta, game, sim = egta_info
    prof = game.random_profile()

    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 25, 0, 0) as sched:
        # so that enough calls to get_requirements are made
        server.throw_exception(TimeoutError)
        await asyncio.sleep(0.1)
        with pytest.raises(TimeoutError):
            await sched.sample_payoffs(prof)


@pytest.mark.asyncio
async def test_scheduler_deactivate(egta_info):
    _, egta, game, sim = egta_info

    # Schedule all new profiles and verify it works
    # This first time should have to wait to schedule more
    async with eosched.EgtaOnlineScheduler(
            game, egta, sim['id'], 1, {}, 0.1, 10, 0, 0) as sched:
        # Deactivate scheduler
        for esched in egta.get_generic_schedulers():
            esched.deactivate()
        with pytest.raises(AssertionError):
            await sched.sample_payoffs(game.random_profile())
