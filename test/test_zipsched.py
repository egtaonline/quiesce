import asyncio
import io
import json
import zipfile

import numpy as np
import pytest
from gameanalysis import rsgame

from egta import countsched
from egta import zipsched


def batch_to_zip(bash):
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w') as zf:
        zf.writestr('package/defaults.json', json.dumps({}))
        zf.writestr('package/script/batch', '#!/usr/bin/env bash\n' + bash)
    return buff


@pytest.fixture
def jgame():
    with open('cdasim/game.json') as f:
        return json.load(f)


@pytest.fixture
def conf(jgame):
    return jgame['configuration']


@pytest.fixture
def game(jgame):
    return rsgame.emptygame_json(jgame)


@pytest.mark.asyncio
async def test_basic_profile(conf, game):
    profs = game.random_profiles(20)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.zipsched(game, conf, zipf) as sched:
        assert str(sched) is not None
        assert rsgame.emptygame_copy(sched) == game
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])

    pays = np.stack(awaited)
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)


@pytest.mark.asyncio
async def test_no_conf(game):
    profs = game.random_profiles(20)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.zipsched(game, {}, zipf) as sched:
        assert rsgame.emptygame_copy(sched) == game
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])

    pays = np.stack(awaited)
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [2, 3])
async def test_simultaneous_obs(conf, game, count):
    profs = game.random_profiles(10).repeat(2, 0)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.zipsched(
            game, conf, zipf, simultaneous_obs=count) as sched:
        sched = countsched.CountScheduler(sched, count)
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])

    pays = np.stack(awaited)
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)


@pytest.mark.asyncio
async def test_get_fail(conf, game):
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2 && false')

    with zipsched.zipsched(game, conf, zipf) as sched:
        future = sched.sample_payoffs(prof)
        with pytest.raises(AssertionError):
            await future


@pytest.mark.asyncio
async def test_bad_zipfile(conf, game):
    zipf = 'nonexistent'
    with pytest.raises(FileNotFoundError):
        with zipsched.zipsched(game, conf, zipf):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_no_obs(conf, game):
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2')

    with zipsched.zipsched(game, conf, zipf) as sched:
        future = sched.sample_payoffs(prof)
        with pytest.raises(AssertionError):
            await future
