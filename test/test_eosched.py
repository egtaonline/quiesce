import async_generator
import asyncio
import random

import numpy as np
import pytest
from egtaonline import api
from egtaonline import mockserver
from gameanalysis import rsgame

from egta import countsched
from egta import eosched


@pytest.fixture
@async_generator.async_generator
async def egta_info():
    game = rsgame.emptygame([4, 4], [11, 11])
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)
        await async_generator.yield_((server, egta, game, egame['id']))


@pytest.mark.asyncio
async def test_basic_profile(egta_info):
    _, egta, game, game_id = egta_info
    profs = game.random_profiles(20)

    # Schedule all new profiles and verify it works
    async with eosched.eosched(
            game, egta, game_id, 0.1, 1, 10, 0, 0) as sched:
        assert game == rsgame.emptygame_copy(sched)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Schedule old profiles and verify it still works
    async with eosched.eosched(
            game, egta, game_id, 0.1, 1, 10, 0, 0) as sched:
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Schedule two at a time, in two batches
    async with eosched.eosched(
            game, egta, game_id, 0.1, 2, 10, 0, 0) as base_sched:
        sched = countsched.CountScheduler(base_sched, 2)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)

    # Try again now that everything should be scheduled
    async with eosched.eosched(
            game, egta, game_id, 0.1, 2, 10, 0, 0) as base_sched:
        sched = countsched.CountScheduler(base_sched, 2)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
    pays = np.stack(awaited)
    assert np.allclose(pays[profs == 0], 0)


def _raise(ex):
    raise ex


@pytest.mark.asyncio
async def test_exception_in_create(egta_info):
    server, egta, game, game_id = egta_info
    server.custom_response(lambda: _raise(TimeoutError))
    with pytest.raises(TimeoutError):
        async with eosched.eosched(
                game, egta, game_id, 0.1, 1, 25, 0, 0):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_exception_in_get(egta_info):
    server, egta, game, game_id = egta_info
    profs = game.random_profiles(20)

    async with eosched.eosched(
            game, egta, game_id, 0.1, 1, 10, 0, 0) as sched:
        futures = asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
        await asyncio.sleep(0.1)
        server.custom_response(lambda: _raise(TimeoutError))
        await asyncio.sleep(0.1)
        with pytest.raises(TimeoutError):
            await futures


@pytest.mark.asyncio
async def test_exception_in_schedule(egta_info):
    server, egta, game, game_id = egta_info
    prof = game.random_profile()

    async with eosched.eosched(
            game, egta, game_id, 0.1, 1, 25, 0, 0) as sched:
        # so that enough calls to get_requirements are made
        server.custom_response(lambda: _raise(TimeoutError))
        await asyncio.sleep(0.1)
        with pytest.raises(TimeoutError):
            await sched.sample_payoffs(prof)


@pytest.mark.asyncio
async def test_scheduler_deactivate(egta_info):
    _, egta, game, game_id = egta_info

    # Schedule all new profiles and verify it works
    # This first time should have to wait to schedule more
    async with eosched.eosched(
            game, egta, game_id, 0.1, 1, 10, 0, 0) as sched:
        # Deactivate scheduler
        async for esched in egta.get_generic_schedulers():
            await esched.deactivate()
        with pytest.raises(AssertionError):
            await sched.sample_payoffs(game.random_profile())
