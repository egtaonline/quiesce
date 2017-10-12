import json
import pytest
import subprocess
import tempfile
from os import path

DIR = path.dirname(path.realpath(__file__))
EGTA = path.join(DIR, '..', 'bin', 'egta')
SIM_DIR = path.join(DIR, '..', 'cdasim')
TINY_GAME = path.join(SIM_DIR, 'tiny_game.json')
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


def test_sim_delayed_fail():
    assert not run('--game-json', SMALL_GAME, 'brute', 'sim', '--', 'bash',
                   '-c', 'sleep 1 && false')


def test_sim_read_delayed_fail():
    assert not run('--game-json', SMALL_GAME, 'brute', 'sim', '--', 'bash',
                   '-c', 'read line && false')


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


@pytest.mark.egta
def test_game_id_brute_egta_game():
    assert run('-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'egta',
               '-m2048', '-t60')


@pytest.mark.egta
def test_game_id_brute_egta_game_wconf():
    with open(SMALL_GAME) as f:
        conf = json.load(f)['configuration']
    with tempfile.NamedTemporaryFile('w') as conf_file:
        json.dump(conf, conf_file)
        conf_file.flush()
        assert run('-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'egta',
                   '-c', conf_file.name, '-m2048', '-t60')
