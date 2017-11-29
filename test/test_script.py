import itertools
import json
import pytest
import subprocess
import tempfile
from os import path

from gameanalysis import rsgame


DIR = path.dirname(path.realpath(__file__))
EGTA = path.join(DIR, '..', 'bin', 'egta')
SIM_DIR = path.join(DIR, '..', 'cdasim')
SMALL_GAME = path.join(SIM_DIR, 'small_game.json')
GAME = path.join(SIM_DIR, 'game.json')
DATA_GAME = path.join(SIM_DIR, 'data_game.json')
SIM = [path.join(DIR, '..', 'bin', 'python'), path.join(SIM_DIR, 'sim.py'),
       '1', '--single']


def run(inp, *args):
    res = subprocess.run((EGTA,) + args, input=inp.encode('utf-8'),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (not res.returncode, res.stdout.decode('utf-8'),
            res.stderr.decode('utf-8'))


def assert_term(sleep1, sleep2, *args):
    proc = subprocess.Popen((EGTA,) + args)
    try:
        proc.wait(sleep1)
        proc.kill()
        assert False, "process finished during first sleep"
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        assert proc.wait(sleep2), "process succeeded after termination"
    except subprocess.TimeoutExpired:
        proc.kill()
        assert False, "process didn't terminate in second sleep"


def test_help():
    succ, _, _ = run('')
    assert not succ
    succ, _, _ = run('', '--fail')
    assert not succ
    succ, _, err = run('', '--help')
    assert succ, err
    succ, _, err = run('', 'brute', '--help')
    assert succ, err
    succ, _, err = run('', 'brute', 'game', '--help')
    assert succ, err
    succ, _, err = run('', 'brute', 'sim', '--help')
    assert succ, err
    succ, _, err = run('', 'brute', 'eo', '--help')
    assert succ, err
    succ, _, err = run('', 'quiesce', '--help')
    assert succ, err


def test_brute_game():
    succ, out, err = run(
        '', '--count', '2', '--game-json', DATA_GAME, 'brute', 'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_brute_game_tag():
    succ, out, err = run(
        '', '--tag', 'test', '--count', '2', '--game-json', DATA_GAME, 'brute',
        'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_brute_game_subgame():
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    sub = json.dumps(reader.to_subgame_json(reader.random_subgames()))
    succ, out, err = run(
        sub, '--count', '2', '--game-json', DATA_GAME, 'brute', '--subgame',
        '-', 'game')
    assert succ, err
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_brute_game_term():
    assert_term(0.5, 0.5, '--count', '2', '--game-json', DATA_GAME, 'brute',
                'game')


def test_brute_dpr_game():
    succ, out, err = run(
        '', '--count', '2', '--game-json', DATA_GAME, 'brute', '--dpr',
        'buyers:2,sellers:2', 'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_brute_hr_game():
    succ, out, err = run(
        '', '--count', '2', '--game-json', DATA_GAME, 'brute', '--hr',
        'buyers:2,sellers:2', 'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_prof_data():
    succ, out, err = run(
        '', '--profile-data', '/dev/null', '--game-json', DATA_GAME, 'brute',
        'game', '--sample')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_sim():
    succ, out, err = run(
        '', '--game-json', SMALL_GAME, 'brute', 'sim', '--', *SIM)
    assert succ, err
    with open(SMALL_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_brute_sim_term():
    assert_term(0.5, 0.5, '--game-json', SMALL_GAME, 'brute', 'sim', '--',
                *SIM)


def test_sim_delayed_fail():
    succ, _, _ = run(
        '', '--game-json', SMALL_GAME, 'brute', 'sim', '--', 'bash', '-c',
        'sleep 1 && false')
    assert not succ


def test_sim_read_delayed_fail():
    succ, _, _ = run(
        '', '--game-json', SMALL_GAME, 'brute', 'sim', '--', 'bash', '-c',
        'read line && false')
    assert not succ


def test_sim_conf():
    with open(SMALL_GAME) as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    conf['markup'] = 'standard'
    reader = rsgame.emptygame_json(jgame)
    with tempfile.NamedTemporaryFile('w') as conf_file:
        json.dump(conf, conf_file)
        conf_file.flush()
        succ, out, err = run(
            '', '--game-json', SMALL_GAME, 'brute', 'sim', '-c',
            conf_file.name, '--', *SIM)
    assert succ, err
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_innerloop():
    succ, out, err = run('', '--game-json', DATA_GAME, 'quiesce', 'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_innerloop_game_term():
    assert_term(0.5, 0.5, '--game-json', DATA_GAME, 'quiesce', 'game')


def test_innerloop_sim_term():
    assert_term(0.5, 0.5, '--game-json', SMALL_GAME, 'quiesce', 'sim', '--',
                *SIM)


def test_innerloop_dpr():
    succ, out, err = run(
        '', '--game-json', DATA_GAME, 'quiesce', '--dpr', 'buyers:2,sellers:2',
        'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


def test_innerloop_hr():
    succ, out, err = run(
        '', '--game-json', DATA_GAME, 'quiesce', '--hr', 'buyers:2,sellers:2',
        'game')
    assert succ, err
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


@pytest.mark.egta
def test_game_id_brute_egta_game():
    succ, out, err = run(
        '', '-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'eo', '2048',
        '60')
    assert succ, err
    with open(SMALL_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    for eqm in json.loads(out):
        reader.from_mix_json(eqm['equilibrium'])


@pytest.mark.egta
def test_game_id_brute_egta_game_wconf():
    with open(SMALL_GAME) as f:
        jgame = json.load(f)
    conf = jgame['configuration']
    reader = rsgame.emptygame_json(jgame)
    with tempfile.NamedTemporaryFile('w') as conf_file:
        json.dump(conf, conf_file)
        conf_file.flush()
        succ, out, err = run(
            '', '-g1466', 'brute', '--dpr', 'buyers:2,sellers:2', 'eo', '-c',
            conf_file.name, '2048', '60')
    assert succ, err
    for eqm in out[:-1].split('\n'):
        reader.from_mix_json(json.loads(eqm))


def test_boot_game():
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    mix = reader.random_mixtures()
    with tempfile.NamedTemporaryFile('w') as mix_file:
        json.dump(reader.to_mix_json(mix), mix_file)
        mix_file.flush()
        succ, out, err = run(
            '', '--game-json', DATA_GAME, 'boot', mix_file.name, '10',
            'game')
    assert succ, err
    results = json.loads(out)
    assert {'total', 'buyers', 'sellers'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()


def test_boot_game_percs():
    with open(DATA_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    mix = reader.random_mixtures()
    succ, out, err = run(
        json.dumps(reader.to_mix_json(mix)), '--game-json', DATA_GAME,
        'boot', '-', '20', '--percentiles', '95,99', 'game')
    assert succ, err
    results = json.loads(out)
    assert {'mean', '99', '95'} == results.keys()
    for val in results.values():
        assert {'total', 'buyers', 'sellers'} == val.keys()
    for val in results['mean'].values():
        assert {'surplus', 'regret', 'response'} == val.keys()
    for val in itertools.chain(results['95'].values(), results['99'].values()):
        assert {'surplus', 'regret'} == val.keys()


def test_boot_sim():
    with open(SMALL_GAME) as f:
        reader = rsgame.emptygame_json(json.load(f))
    mix = reader.random_mixtures()
    succ, out, err = run(
        json.dumps(reader.to_mix_json(mix)), '--game-json', SMALL_GAME,
        'boot', '-', '50', '--chunk-size', '10', 'sim', '--', *SIM)
    assert succ, err
    results = json.loads(out)
    assert {'total', 'buyers', 'sellers'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()
