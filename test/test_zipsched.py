import io
import json
import pytest
import time
import zipfile

import numpy as np
from gameanalysis import rsgame

from egta import zipsched


def batch_to_zip(bash):
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w') as zf:
        zf.writestr('package/defaults.json', json.dumps({}))
        zf.writestr('package/script/batch', '#!/usr/bin/env bash\n' + bash)
    return buff


def test_basic_profile():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game = rsgame.emptygame_json(jgame)
    profs = game.random_profiles(20)
    zipf = 'cdasim/cdasim.zip'

    with zipsched.ZipScheduler(game, conf, zipf) as sched:
        assert rsgame.emptygame_copy(sched.game()) == game
        proms = [sched.schedule(p) for p in profs]
        pays = np.stack([p.get() for p in proms])
    assert pays.shape == profs.shape
    assert np.allclose(pays[profs == 0], 0)
    assert np.any(pays != 0)


def test_get_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game = rsgame.emptygame_json(jgame)
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2 && false')

    with zipsched.ZipScheduler(game, conf, zipf) as sched:
        prom = sched.schedule(prof)
        with pytest.raises(AssertionError):
            prom.get()


def test_schedule_fail():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game = rsgame.emptygame_json(jgame)
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2 && false')

    with zipsched.ZipScheduler(game, conf, zipf) as sched:
        sched.schedule(prof)
        time.sleep(1)
        with pytest.raises(AssertionError):
            sched.schedule(prof)


def test_kill_extra_procs():
    with open('cdasim/game.json') as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    game = rsgame.emptygame_json(jgame)
    prof = game.random_profile()
    zipf = batch_to_zip('sleep 0.2 && false')

    with zipsched.ZipScheduler(game, conf, zipf) as sched:
        prom = sched.schedule(prof)
        sched.schedule(prof)
        time.sleep(1)
        with pytest.raises(AssertionError):
            prom.get()
