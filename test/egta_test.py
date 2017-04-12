import json
import os
import pytest
import subprocess
import tempfile
from os import path

from gameanalysis import gameio


DIR = path.dirname(path.realpath(__file__))
EGTA = path.join(DIR, '..', 'bin', 'egta')
SIM_DIR = path.join(DIR, '..', 'cdasim')
SMALL_GAME = path.join(SIM_DIR, 'small_game.json')
GAME = path.join(SIM_DIR, 'game.json')
DATA_GAME = path.join(SIM_DIR, 'data_game.json')
SIM = [path.join(DIR, '..', 'bin', 'python'), path.join(SIM_DIR, 'sim.py'),
       '1', '--single']


def run(*args):
    return not subprocess.run((EGTA,) + args).returncode


def test_help():
    assert not run('--fail')
    assert not run()
    assert run('--help')
    assert run('brute', '--help')
    assert run('brute', 'game', '--help')
    assert run('brute', 'sim', '--help')
    assert run('brute', 'egta', '--help')
    assert run('quiesce', '--help')


def test_brute_game():
    assert not run('--game-json', DATA_GAME, 'brute', 'game', '--load-game',
                   '--load-samplegame')
    assert run('--count', '2', '--game-json', DATA_GAME, 'brute', 'game',
               '--load-game')


def test_brute_dpr_game():
    assert run('--count', '2', '--game-json', DATA_GAME, 'brute', '--dpr',
               'buyers:2,sellers:2', 'game', '--load-game')


def test_prof_data():
    assert run('--profile-data', '/dev/null', '--game-json', DATA_GAME,
               'brute', 'game', '--load-samplegame')


def test_sim():
    assert run('--game-json', SMALL_GAME, 'brute', 'sim', '--', *SIM)


def test_sim_conf():
    with open(SMALL_GAME) as f:
        conf = json.load(f)['configuration']
    conf['markup'] = 'standard'
    with tempfile.NamedTemporaryFile('w') as conf_file:
        json.dump(conf, conf_file)
        conf_file.flush()
        assert run('--game-json', SMALL_GAME, 'brute', 'sim', '-c',
                   conf_file.name, '--', *SIM)


def test_innerloop():
    assert run('--game-json', DATA_GAME, 'quiesce', 'game', '--load-game')


def test_innerloop_dpr():
    assert run('--game-json', DATA_GAME, 'quiesce', '--dpr',
               'buyers:2,sellers:2', 'game', '--load-game')


def test_outerloop():
    with open(DATA_GAME) as f:
        bgame, serial = gameio.read_basegame(json.load(f))
    sub_json = serial.to_prof_json(bgame.random_subgames())
    with tempfile.NamedTemporaryFile('w') as sub_file:
        json.dump(sub_json, sub_file)
        sub_file.flush()
        assert run('--game-json', DATA_GAME, 'quiesce', '--initial-subgame',
                   sub_file.name, 'game', '--load-game')


@pytest.mark.skipif('EGTA_TESTS' not in os.environ,
                    reason="Don't run egta tests by default")
def test_game_id_brute_egta_game():
    assert run('-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'egta',
               '-m2048', '-t60')


@pytest.mark.skipif('EGTA_TESTS' not in os.environ,
                    reason="Don't run egta tests by default")
def test_game_id_brute_egta_game_wconf():
    with open(SMALL_GAME) as f:
        conf = json.load(f)['configuration']
    with tempfile.NamedTemporaryFile('w') as conf_file:
        json.dump(conf, conf_file)
        conf_file.flush()
        assert run('-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'egta',
                   '-c', conf_file.name, '-m2048', '-t60')
