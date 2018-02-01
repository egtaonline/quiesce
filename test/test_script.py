import io
import itertools
import json
import random
import subprocess
import sys
import traceback
from os import path
from unittest import mock

import numpy as np
import pytest
from egtaonline import api
from egtaonline import mockserver
from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import rsgame
from gameanalysis import utils as gu

from egta import __main__ as main
from test import utils as tu


base = path.dirname(path.realpath(__file__))
DIR = base
EGTA = path.join(DIR, '..', 'bin', 'egta')
SIM_DIR = path.join(DIR, '..', 'cdasim')


def run(*args):
    """Run a command line and return if it ran successfully"""
    try:
        main.main(*args)
    except SystemExit as ex:
        return not int(str(ex))
    except Exception:
        traceback.print_exc()
        return False
    return True


def stdin(inp):
    """Patch stdin with input"""
    return mock.patch.object(sys, 'stdin', io.StringIO(inp))


def stdout():
    """Patch stdout and return stringio"""
    return mock.patch.object(sys, 'stdout', io.StringIO())


def stderr():
    """Patch stderr and return stringio"""
    return mock.patch.object(sys, 'stderr', io.StringIO())


@pytest.fixture(scope='session')
def game_info(tmpdir_factory):
    game_file = str(tmpdir_factory.mktemp('games').join('game.json'))
    game = gamegen.game([3, 2], [2, 3])
    with open(game_file, 'w') as f:
        json.dump(game.to_json(), f)
    return (game, game_file)


@pytest.fixture
def sim_game_info():
    sim_dir = path.join(base, '..', 'cdasim')
    game_file = path.join(sim_dir, 'small_game.json')
    sim = [path.join(base, '..', 'bin', 'python'),
           path.join(sim_dir, 'sim.py'), '1', '--single']
    zip_file = path.join(sim_dir, 'cdasim.zip')
    with open(game_file) as f:
        jgame = json.load(f)
    return (gamereader.loadj(jgame), game_file, sim, zip_file,
            jgame['configuration'])


def assert_term(sleep1, sleep2, *args):
    proc = subprocess.Popen((EGTA,) + args)
    try:
        proc.wait(sleep1)
        # proc.wait should throw exception and pass i.e. process is still
        # running after sleep1
        proc.kill()  # pragma: no cover
        assert False, "process finished during first sleep"  # pragma: no cover
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        assert proc.wait(sleep2), "process succeeded after termination"
    except subprocess.TimeoutExpired:  # pragma: no cover
        # proc.terminate should terminate process before sleep2 is reached, so
        # we should never trigger except
        proc.kill()
        assert False, "process didn't terminate in second sleep"


def test_help():
    assert not run()
    assert not run('--fail')
    with stderr() as err:
        assert run('--help'), err.getvalue()
    with stderr() as err:
        assert run('brute', '--help'), err.getvalue()
    with stderr() as err:
        assert run('quiesce', '--help'), err.getvalue()
    with stderr() as err:
        assert run('boot', '--help'), err.getvalue()
    with stderr() as err:
        assert run('trace', '--help'), err.getvalue()
    with stderr() as err:
        assert run('brute', 'game', '--help'), err.getvalue()
    with stderr() as err:
        assert run('brute', 'sim', '--help'), err.getvalue()
    with stderr() as err:
        assert run('brute', 'zip', '--help'), err.getvalue()
    with stderr() as err:
        assert run('brute', 'eo', '--help'), err.getvalue()


def test_brute_game(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--count', '2', '--game-json', game_file, 'brute',
            'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_game_tag(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--tag', 'test', '--count', '2', '--game-json', game_file, 'brute',
            'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_game_restriction(game_info):
    game, game_file = game_info
    rest = json.dumps(game.restriction_to_json(game.random_restriction()))
    with stdin(rest), stdout() as out, stderr() as err:
        assert run(
            '--count', '2', '--game-json', game_file, 'brute', '--restrict',
            '-', 'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_game_term(game_info):
    assert_term(0.5, 0.5, '--count', '2', '--game-json', game_info[1], 'brute',
                'game')


def test_brute_dpr_game(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--count', '2', '--game-json', game_file, 'brute', '--dpr',
            'r0:2;r1:2', 'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_hr_game(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--count', '2', '--game-json', game_file, 'brute', '--hr',
            'r0:2;r1:2', 'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_prof_data(game_info, tmpdir):
    game, game_file = game_info
    prof_file = str(tmpdir.join('profs.json'))
    with stdout() as out, stderr() as err:
        assert run(
            '--profile-data', prof_file, '--game-json', game_file, 'brute',
            'game', '--sample'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])
    with open(prof_file) as f:
        prof_game = gamereader.load(f)
    assert rsgame.emptygame_copy(game) == rsgame.emptygame_copy(prof_game)


def test_sim(sim_game_info):
    game, game_file, sim, *_ = sim_game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'brute', '--dpr', 'buyers:2;sellers:2',
            'sim', '--', *sim), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_zip(sim_game_info):
    game, game_file, _, zip_file, _ = sim_game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'brute', '--dpr', 'buyers:2;sellers:2',
            'zip', zip_file, '-p1'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_sim_term(sim_game_info):
    _, game_file, sim, *_ = sim_game_info
    assert_term(0.5, 0.5, '--game-json', game_file, 'brute', 'sim', '--', *sim)


def test_sim_delayed_fail(game_info):
    with stdin(json.dumps({})):
        assert not run(
            '--game-json', game_info[1], 'brute', 'sim', '--conf', '-', '--',
            'bash', '-c', 'sleep 1 && false')


def test_sim_conf(sim_game_info, tmpdir):
    game, game_file, sim, _, conf = sim_game_info
    conf_file = str(tmpdir.join('conf.json'))
    with open(conf_file, 'w') as f:
        json.dump(dict(conf, markup='standard'), f)
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'brute', '--dpr', 'buyers:2;sellers:2',
            'sim', '-c', conf_file, '--', *sim), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@tu.warnings_filter(RuntimeWarning)
def test_innerloop(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run('--game-json', game_file, 'quiesce', 'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_innerloop_game_term(game_info):
    assert_term(0.5, 0.5, '--game-json', game_info[1], 'quiesce', 'game')


def test_innerloop_sim_term(sim_game_info):
    _, game_file, sim, *_ = sim_game_info
    assert_term(0.5, 0.5, '--game-json', game_file, 'quiesce', 'sim', '--',
                *sim)


def test_innerloop_dpr(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'quiesce', '--dpr', 'r0:2;r1:2',
            'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_innerloop_hr(game_info):
    game, game_file = game_info
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'quiesce', '--hr', 'r0:2;r1:2',
            'game'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def get_gameid(server, game, name):
    with api.EgtaOnlineApi() as egta:
        sim = egta.get_simulator(server.create_simulator(
            name, '1', delay_dist=lambda: random.random() / 100))
        sim.add_dict({role: strats for role, strats
                      in zip(game.role_names, game.strat_names)})
        eogame = sim.create_game(name, game.num_players)
        for role, count, strats in zip(game.role_names, game.num_role_players,
                                       game.strat_names):
            eogame.add_role(role, count)
            for strat in strats:
                eogame.add_strategy(role, strat)
        return sim['id'], eogame['id']


def test_game_id_brute_egta_game(game_info):
    game, game_file = game_info
    with mockserver.Server() as server:
        _, game_id = get_gameid(server, game, 'game')
        with stdout() as out, stderr() as err:
            assert run(
                '-g', str(game_id), 'brute', '--dpr', 'r0:2;r1:2', 'eo',
                '2048', '60', '--sleep', '0.1'), err.getvalue()
            for eqm in json.loads(out.getvalue()):
                game.mixture_from_json(eqm['equilibrium'])


def test_game_id_brute_egta_game_wconf(game_info):
    game, game_file = game_info
    conf = json.dumps({})
    with mockserver.Server() as server:
        sim_id, _ = get_gameid(server, game, 'game')
        with stdin(conf), stdout() as out, stderr() as err:
            assert run(
                '--game-json', game_file, 'brute', '--dpr',
                'r0:2;r1:2', 'eo', '-c', '-', '2048', '60', '--sleep',
                '0.1', '--sim-id', str(sim_id)), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def verify_trace_json(game, traces):
    for trace in traces:
        for point in trace:
            assert point.keys() == {'regret', 'equilibrium', 't'}
            game.mixture_from_json(point['equilibrium'])
        assert gu.is_sorted(trace, key=lambda t: t['t'])


@tu.warnings_filter(RuntimeWarning)
def test_trace_dpr(tmpdir):
    game = rsgame.emptygame([3, 2], [2, 3])
    prof_data = str(tmpdir.join('data'))
    with mockserver.Server() as server:
        _, game1 = get_gameid(server, game, 'game1')
        _, game2 = get_gameid(server, game, 'game2')
        with stdout() as out, stderr() as err:
            assert run(
                '-g', str(game1), '--profile-data', prof_data, 'trace',
                str(game2), '2048', '60', '--dpr', 'r0:2;r1:2', 'eo', '2048',
                '60', '--sleep', '1'), err.getvalue()
        traces = json.loads(out.getvalue())
    verify_trace_json(game, traces)
    with open(prof_data) as f:
        data = gamereader.load(f)
    assert game == rsgame.emptygame_copy(data)


@tu.warnings_filter(RuntimeWarning)
def test_trace_hr():
    game = rsgame.emptygame([3, 2], [2, 3])
    with mockserver.Server() as server:
        _, game1 = get_gameid(server, game, 'game1')
        _, game2 = get_gameid(server, game, 'game2')
        with stdout() as out, stderr() as err:
            assert run(
                '-g', str(game1), '--count', '2', 'trace', str(game2), '2048',
                '60', '--hr', 'r0:2;r1:2', 'eo', '2048', '60', '--sleep',
                '1'), err.getvalue()
        traces = json.loads(out.getvalue())
    verify_trace_json(game, traces)


def add_singleton_role(game, index, role_name, strat_name, num_players):
    """Add a singleton role to a game"""
    role_names = list(game.role_names)
    role_names.insert(index, role_name)
    strat_names = list(game.strat_names)
    strat_names.insert(index, [strat_name])
    role_players = np.insert(game.num_role_players, index, num_players)
    return rsgame.emptygame_names(role_names, role_players, strat_names)


@tu.warnings_filter(RuntimeWarning)
def test_trace_norm():
    base_game = rsgame.emptygame([3, 2], [2, 3])
    with mockserver.Server() as server:
        game1 = add_singleton_role(base_game, 0, 'a', 'sx', 1)
        _, id1 = get_gameid(server, game1, 'game1')
        game2 = add_singleton_role(base_game, 1, 'r00', 's0', 3)
        _, id2 = get_gameid(server, game2, 'game2')
        with stdout() as out, stderr() as err:
            assert run(
                '-g', str(id1), 'trace', str(id2), '2048', '60', 'eo', '2048',
                '60', '--sleep', '1'), err.getvalue()
        traces = json.loads(out.getvalue())
    verify_trace_json(base_game, traces)


def test_boot_game(game_info, tmpdir):
    game, game_file = game_info
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'boot', mix_file, '10',
            'game'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'total', 'r0', 'r1'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()


def test_boot_symmetric(tmpdir):
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert run(
            '--game-json', '-', 'boot', mix_file, '10', 'game'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'surplus', 'regret', 'response'} == results.keys()


def test_boot_symmetric_percs(tmpdir):
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert run(
            '--game-json', '-', 'boot', mix_file, '10', '-p', '95',
            'game'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'95', 'mean'} == results.keys()
    assert {'surplus', 'regret', 'response'} == results['mean'].keys()
    assert {'surplus', 'regret'} == results['95'].keys()


def test_boot_game_percs(game_info):
    game, game_file = game_info
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'boot', '-', '20', '--percentile', '95',
            '-p99', 'game'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'mean', '99', '95'} == results.keys()
    for val in results.values():
        assert {'total', 'r0', 'r1'} == val.keys()
    for val in results['mean'].values():
        assert {'surplus', 'regret', 'response'} == val.keys()
    for val in itertools.chain(results['95'].values(), results['99'].values()):
        assert {'surplus', 'regret'} == val.keys()


def test_boot_sim(sim_game_info):
    game, game_file, sim, *_ = sim_game_info
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert run(
            '--game-json', game_file, 'boot', '-', '50', '--chunk-size', '10',
            'sim', '--', *sim), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'total', 'buyers', 'sellers'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()
