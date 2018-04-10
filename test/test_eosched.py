import asyncio
import random

import numpy as np
import pytest
from egtaonline import api
from egtaonline import mockserver
from gameanalysis import rsgame

from egta import countsched
from egta import eosched


# TODO general setup may be done with a fixture in python 3.6

@pytest.fixture
def game():
    return rsgame.emptygame([4, 4], [11, 11])


@pytest.mark.asyncio
async def test_basic_profile(game):
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)

        profs = game.random_profiles(20)

        # Schedule all new profiles and verify it works
        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 1, 10, 0, 0) as sched:
            assert str(sched) == str(egame['id'])
            assert game == rsgame.emptygame_copy(sched)
            awaited = await asyncio.gather(*[
                sched.sample_payoffs(p) for p in profs])
        pays = np.stack(awaited)
        assert np.allclose(pays[profs == 0], 0)

        # Schedule old profiles and verify it still works
        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 1, 10, 0, 0) as sched:
            awaited = await asyncio.gather(*[
                sched.sample_payoffs(p) for p in profs])
        pays = np.stack(awaited)
        assert np.allclose(pays[profs == 0], 0)

        # Schedule two at a time, in two batches
        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 2, 10, 0, 0) as base_sched:
            sched = countsched.CountScheduler(base_sched, 2)
            awaited = await asyncio.gather(*[
                sched.sample_payoffs(p) for p in profs])
        pays = np.stack(awaited)
        assert np.allclose(pays[profs == 0], 0)

        # Try again now that everything should be scheduled
        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 2, 10, 0, 0) as base_sched:
            sched = countsched.CountScheduler(base_sched, 2)
            awaited = await asyncio.gather(*[
                sched.sample_payoffs(p) for p in profs])
        pays = np.stack(awaited)
        assert np.allclose(pays[profs == 0], 0)


def _raise(ex):
    raise ex


@pytest.mark.asyncio
async def test_exception_in_create(game):
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(  # pragma: no branch # noqa
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)

        server.custom_response(lambda: _raise(TimeoutError))
        with pytest.raises(TimeoutError):
            async with eosched.eosched(
                    game, egta, egame['id'], 0.1, 1, 25, 0, 0):
                pass  # pragma: no cover


@pytest.mark.asyncio
async def test_exception_in_get(game):
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)

        profs = game.random_profiles(20)

        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 1, 10, 0, 0) as sched:
            futures = asyncio.gather(*[
                sched.sample_payoffs(p) for p in profs])
            await asyncio.sleep(0.1)
            server.custom_response(lambda: _raise(TimeoutError))
            await asyncio.sleep(0.1)
            with pytest.raises(TimeoutError):
                await futures


@pytest.mark.asyncio
async def test_exception_in_schedule(game):
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(  # pragma: no branch # noqa
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)

        prof = game.random_profile()

        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 1, 25, 0, 0) as sched:
            # so that enough calls to get_requirements are made
            server.custom_response(lambda: _raise(TimeoutError))
            await asyncio.sleep(0.1)
            with pytest.raises(TimeoutError):
                await sched.sample_payoffs(prof)


@pytest.mark.asyncio
async def test_scheduler_deactivate(game):
    async with mockserver.server() as server, api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(  # pragma: no branch # noqa
            'sim', '1', delay_dist=lambda: random.random() / 10))
        strats = dict(zip(game.role_names, game.strat_names))
        symgrps = list(zip(game.role_names, game.num_role_players,
                           game.strat_names))
        await sim.add_strategies(strats)
        egame = await egta.get_canon_game(sim['id'], symgrps)

        # Schedule all new profiles and verify it works
        # This first time should have to wait to schedule more
        async with eosched.eosched(
                game, egta, egame['id'], 0.1, 1, 10, 0, 0) as sched:
            # Deactivate scheduler
            for esched in await egta.get_generic_schedulers():
                await esched.deactivate()
            with pytest.raises(ValueError):
                await sched.sample_payoffs(game.random_profile())
