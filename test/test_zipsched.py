"""Test zip scheduler"""
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
    """Convert the string of a bash script into a zip file"""
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w') as zfil:
        zfil.writestr('package/defaults.json', json.dumps({}))
        zfil.writestr('package/script/batch', '#!/usr/bin/env bash\n' + bash)
    return buff


@pytest.fixture(name='jgame')
def fix_jgame():
    """Fixture for the common game json"""
    with open('cdasim/game.json') as fil:
        return json.load(fil)


@pytest.fixture(name='conf')
def fix_conf(jgame):
    """Fixture for a default configuration"""
    return jgame['configuration']


@pytest.fixture(name='game')
def fix_game(jgame):
    """Fixture a game"""
    return rsgame.empty_json(jgame)


@pytest.mark.asyncio
async def test_basic_profile(conf, game):
    """Test scheduling a basic profile"""
    profs = game.random_profiles(20)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.zipsched(game, conf, zipf) as sched:
        assert str(sched) is not None
        assert rsgame.empty_copy(sched) == game
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])

    pays = np.stack(awaited)
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)
    # XXX This line exists to fool duplication check 1


@pytest.mark.asyncio
async def test_no_conf(game):
    """Test scheduler without a configuration"""
    profs = game.random_profiles(20)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.zipsched(game, {}, zipf) as sched:
        assert rsgame.empty_copy(sched) == game
        awaited = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])

    pays = np.stack(awaited)
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)
    # XXX This line exists to fool duplication check 2


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [2, 3])
async def test_simultaneous_obs(conf, game, count):
    """Test scheduler appropriately schedules simultaneous observations"""
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
    # XXX This line exists to fool duplication check 3


@pytest.mark.asyncio
async def test_get_fail(conf, game):
    """Test zip file that fails"""
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2 && false')

    with zipsched.zipsched(game, conf, zipf) as sched:
        future = sched.sample_payoffs(prof)
        with pytest.raises(ValueError):
            await future


@pytest.mark.asyncio
async def test_bad_zipfile(conf, game):
    """Test that improper zip file fails"""
    zipf = 'nonexistent'
    with pytest.raises(FileNotFoundError):
        with zipsched.zipsched(game, conf, zipf):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_no_obs(conf, game):
    """Test zip sched that doesn't write observations"""
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2')

    with zipsched.zipsched(game, conf, zipf) as sched:
        future = sched.sample_payoffs(prof)
        with pytest.raises(ValueError):
            await future
