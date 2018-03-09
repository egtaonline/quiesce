import asyncio

import numpy as np
import numpy.random as rand
import pytest
from gameanalysis import agggen
from gameanalysis import gamegen
from gameanalysis import rsgame

from egta import gamesched


@pytest.mark.asyncio
async def test_basic_profile():
    game = gamegen.game([4, 3], [3, 4])
    profs = game.random_profiles(20)

    sched = gamesched.gamesched(game)
    assert (rsgame.emptygame_copy(sched) ==
            rsgame.emptygame_copy(game))
    paylist = await asyncio.gather(*[
        sched.sample_payoffs(p) for p in profs])
    pays = np.stack(paylist)
    assert np.allclose(pays[profs == 0], 0)


@pytest.mark.asyncio
async def test_basic_profile_sample():
    sgame = gamegen.samplegame([4, 3], [3, 4])
    profs = sgame.random_profiles(20)

    sched = gamesched.samplegamesched(sgame)
    assert (rsgame.emptygame_copy(sched) ==
            rsgame.emptygame_copy(sgame))
    paylist = await asyncio.gather(*[
        sched.sample_payoffs(p) for p in profs])
    pays = np.stack(paylist)
    assert np.allclose(pays[profs == 0], 0)


@pytest.mark.asyncio
async def test_duplicate_profile_sample():
    sgame = gamegen.samplegame([4, 3], [3, 4], 0)
    profs = sgame.random_profiles(20)

    sched = gamesched.samplegamesched(sgame)
    paylist1 = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays1 = np.stack(paylist1)
    paylist2 = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays2 = np.stack(paylist2)
    assert np.allclose(pays1[profs == 0], 0)
    assert np.allclose(pays2[profs == 0], 0)
    assert np.allclose(pays1, pays2)


@pytest.mark.asyncio
async def test_basic_profile_aggfn():
    agame = agggen.normal_aggfn([4, 3], [3, 4], 5)
    profs = agame.random_profiles(20)

    sched = gamesched.gamesched(agame)
    paylist = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays = np.stack(paylist)
    assert np.allclose(pays[profs == 0], 0)


@pytest.mark.asyncio
async def test_noise_profile():
    sgame = gamegen.samplegame([4, 3], [3, 4])
    profs = sgame.random_profiles(20)

    sched = gamesched.samplegamesched(
        sgame, lambda w: rand.normal(0, w, sgame.num_strats),
        lambda: (rand.random(),))
    paylist = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays = np.stack(paylist)
    assert np.allclose(pays[profs == 0], 0)


@pytest.mark.asyncio
async def test_duplicate_prof():
    game = gamegen.game([4, 3], [3, 4])
    profs = game.random_profiles(20)

    sched = gamesched.gamesched(game)
    paylist1 = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays1 = np.stack(paylist1)
    paylist2 = await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    pays2 = np.stack(paylist2)
    assert np.allclose(pays1[profs == 0], 0)
    assert np.allclose(pays2[profs == 0], 0)
    assert np.allclose(pays1, pays2)
