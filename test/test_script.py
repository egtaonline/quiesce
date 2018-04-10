import contextlib
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
from gameanalysis import utils

from egta import __main__ as main


base = path.dirname(path.realpath(__file__))
DIR = base
EGTA = path.join(DIR, '..', 'bin', 'egta')
SIM_DIR = path.join(DIR, '..', 'cdasim')


async def run(*args):
    """Run a command line and return if it ran successfully"""
    try:
        await main.amain(*args)
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
    return contextlib.redirect_stdout(io.StringIO())


def stderr():
    """Patch stderr and return stringio"""
    return contextlib.redirect_stderr(io.StringIO())


@pytest.fixture(scope='session')
def game_info(tmpdir_factory):
    game_file = str(tmpdir_factory.mktemp('games').join('game.json'))
    game = gamegen.game([3, 2], [2, 3])
    with open(game_file, 'w') as f:
        json.dump(game.to_json(), f)
    return game, 'game:game:{}'.format(game_file)


@pytest.fixture(scope='session')
def sim_game_info(tmpdir_factory):
    sim_dir = path.join(base, '..', 'cdasim')
    game_file = path.join(sim_dir, 'small_game.json')
    sim = [path.join(base, '..', 'bin', 'python'),
           path.join(sim_dir, 'sim.py'), '1', '--single']
    zip_file = path.join(sim_dir, 'cdasim.zip')
    with open(game_file) as f:
        jgame = json.load(f)
    conf_file = str(tmpdir_factory.mktemp('conf').join('conf.json'))
    with open(conf_file, 'w') as f:
        json.dump(jgame['configuration'], f)
    game = rsgame.emptygame_copy(gamereader.loadj(jgame))
    simsched = 'sim:game:{},conf:{},command:{}'.format(
        game_file, conf_file, ' '.join(sim))
    zipsched = 'zip:game:{},zipf:{}'.format(game_file, zip_file)
    return game, simsched, zipsched


def assert_term(sleep1, sleep2, *args):
    proc = subprocess.Popen((EGTA,) + args)
    try:
        proc.wait(sleep1)
        # proc.wait should throw exception and pass i.e. process is still
        # running after sleep1
        proc.kill()  # pragma: no cover
        assert False, 'process finished during first sleep'  # pragma: no cover
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        assert proc.wait(sleep2), 'process succeeded after termination'
    except subprocess.TimeoutExpired:  # pragma: no cover
        # proc.terminate should terminate process before sleep2 is reached, so
        # we should never trigger except
        proc.kill()
        assert False, "process didn't terminate in second sleep"


@pytest.mark.asyncio
async def test_help():
    assert not await run()
    assert not await run('--fail')
    with stderr() as err:
        assert await run('--help'), err.getvalue()
    with stderr() as err:
        assert await run('brute', '--help'), err.getvalue()
    with stderr() as err:
        assert await run('quiesce', '--help'), err.getvalue()
    with stderr() as err:
        assert await run('boot', '--help'), err.getvalue()
    with stderr() as err:
        assert await run('trace', '--help'), err.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize('red', [
    [],
    ['--dpr', 'r0:2,r1:2'],
    ['--hr', 'r0:2,r1:2'],
])
async def test_brute_game(game_info, red):
    game, sched = game_info
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',count:2', *red), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_brute_game_tag(game_info):
    game, sched = game_info
    with stdout() as out, stderr() as err:
        assert await run(
            '--tag', 'test', 'brute', sched + ',count:2'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_brute_game_restriction(game_info):
    game, sched = game_info
    rest = json.dumps(game.restriction_to_json(game.random_restriction()))
    with stdin(rest), stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',count:2', '--restrict', '-'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_game_term(game_info):
    _, sched = game_info
    assert_term(0.5, 0.5, 'brute', sched + ',count:2')


@pytest.mark.asyncio
async def test_prof_data(game_info, tmpdir):
    game, sched = game_info
    prof_file = str(tmpdir.join('profs.json'))
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',sample:,save:' + prof_file), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])
    with open(prof_file) as f:
        prof_game = gamereader.load(f)
    assert rsgame.emptygame_copy(game) == rsgame.emptygame_copy(prof_game)


@pytest.mark.asyncio
async def test_sim(sim_game_info):
    game, sched, _ = sim_game_info
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', sched, '--dpr', 'buyers:2;sellers:2',
            '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_sim_in(sim_game_info):
    game, sched, _ = sim_game_info
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',game:-', '--dpr', 'buyers:2;sellers:2',
            '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_zip(sim_game_info):
    game, _, sched = sim_game_info
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', sched, '--dpr', 'buyers:2;sellers:2',
            '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_zip_in(sim_game_info, tmpdir):
    game, _, sched = sim_game_info
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',game:-', '--dpr', 'buyers:2;sellers:2',
            '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_zip_conf_in(sim_game_info):
    game, _, sched = sim_game_info
    conf = json.dumps({'max_value': 2})
    with stdin(conf), stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',conf:-', '--dpr', 'buyers:2;sellers:2',
            '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_zip_conf_file(sim_game_info, tmpdir):
    game, _, sched = sim_game_info
    conf_file = str(tmpdir.join('conf.json'))
    with open(conf_file, 'w') as f:
        json.dump({'max_value': 2}, f)
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', sched + ',conf:' + conf_file, '--dpr',
            'buyers:2;sellers:2', '--min-reg'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_brute_sim_term(sim_game_info):
    _, sched, _ = sim_game_info
    assert_term(0.5, 0.5, 'brute', sched)


@pytest.mark.asyncio
async def test_sim_delayed_fail(tmpdir):
    game = rsgame.emptygame(4, 3)
    game_file = str(tmpdir.join('game.json'))
    with open(game_file, 'w') as f:
        json.dump(game.to_json(), f)
    script = str(tmpdir.join('script.sh'))
    with open(script, 'w') as f:
        f.write('sleep 1 && false')
    sched = 'sim:game:{},command:bash {}'.format(game_file, script)
    with stdin(json.dumps({})):
        assert not await run(
            'brute', sched)


@pytest.mark.asyncio
async def test_sim_conf(sim_game_info, tmpdir):
    game, sched, _ = sim_game_info
    conf_file = str(tmpdir.join('conf.json'))
    with open(conf_file, 'w') as f:
        json.dump({
            'markup': 'standard',
            'max_value': 1,
            'arrivals': 'simple',
            'market': 'call',
        }, f)
    with stdout() as out, stderr() as err:
        assert await run(
            'brute', '--dpr', 'buyers:2;sellers:2', sched + ',conf:' +
            conf_file), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_sim_conf_in(sim_game_info, tmpdir):
    game, sched, _ = sim_game_info
    conf = json.dumps({
        'markup': 'standard',
        'max_value': 1,
        'arrivals': 'simple',
        'market': 'call',
    })
    with stdin(conf), stdout() as out, stderr() as err:
        assert await run(
            'brute', '--dpr', 'buyers:2;sellers:2',
            sched + ',conf:-'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_innerloop(game_info):
    game, sched = game_info
    with stdout() as out, stderr() as err:
        assert await run('quiesce', sched), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


def test_innerloop_game_term(game_info):
    _, sched = game_info
    assert_term(0.5, 0.5, 'quiesce', sched)


def test_innerloop_sim_term(sim_game_info):
    _, sched, _ = sim_game_info
    assert_term(0.5, 0.5, 'quiesce', sched)


@pytest.mark.asyncio
async def test_innerloop_dpr(game_info):
    game, sched = game_info
    with stdout() as out, stderr() as err:
        assert await run(
            'quiesce', sched, '--dpr', 'r0:2;r1:2'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


@pytest.mark.asyncio
async def test_innerloop_hr(game_info):
    game, sched = game_info
    with stdout() as out, stderr() as err:
        assert await run(
            'quiesce', sched, '--hr', 'r0:2;r1:2'), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm['equilibrium'])


async def get_eosched(server, game, name):
    async with api.api() as egta:
        sim = await egta.get_simulator(server.create_simulator(
            name, '1', delay_dist=lambda: random.random() / 100))
        await sim.add_strategies(dict(zip(game.role_names, game.strat_names)))
        eogame = await sim.create_game(name, game.num_players)
        await eogame.add_symgroups(list(zip(
            game.role_names, game.num_role_players, game.strat_names)))
        return 'eo:game:{:d},mem:2048,time:60,sleep:0.1'.format(
            eogame['id'])


@pytest.mark.asyncio
async def test_brute_egta_game(game_info):
    game, game_file = game_info
    async with mockserver.server() as server:
        sched = await get_eosched(server, game, 'game')
        with stdout() as out, stderr() as err:
            assert await run(
                'brute', sched, '--dpr', 'r0:2;r1:2',
                '--min-reg'), err.getvalue()
            for eqm in json.loads(out.getvalue()):
                game.mixture_from_json(eqm['equilibrium'])


def verify_trace_json(game, traces):
    for trace in traces:
        for point in trace:
            assert point.keys() == {'regret', 'equilibrium', 't'}
            game.mixture_from_json(point['equilibrium'])
        assert utils.is_sorted(trace, key=lambda t: t['t'])


@pytest.mark.asyncio
@pytest.mark.parametrize('red', [
    [],
    ['--dpr', 'buyers:2,sellers:2'],
    ['--hr', 'buyers:2,sellers:2'],
])
async def test_trace(sim_game_info, tmpdir, red):
    game, sched, _ = sim_game_info
    conf_file = str(tmpdir.join('conf.json'))
    with open(conf_file, 'w') as f:
        json.dump({
            'markup': 'standard',
            'max_value': 1,
            'arrivals': 'simple',
            'market': 'call',
        }, f)
    prof_data = str(tmpdir.join('data.json'))
    with stdout() as out, stderr() as err:
        assert await run(
            'trace', sched + ',save:' + prof_data, sched + ',conf:' +
            conf_file, *red), err.getvalue()
    traces = json.loads(out.getvalue())
    verify_trace_json(game, traces)
    with open(prof_data) as f:
        data = gamereader.load(f)
    assert game == rsgame.emptygame_copy(data)


def add_singleton_role(game, index, role_name, strat_name, num_players):
    """Add a singleton role to a game"""
    role_names = list(game.role_names)
    role_names.insert(index, role_name)
    strat_names = list(game.strat_names)
    strat_names.insert(index, [strat_name])
    role_players = np.insert(game.num_role_players, index, num_players)
    return rsgame.emptygame_names(role_names, role_players, strat_names)


@pytest.mark.asyncio
async def test_trace_norm():
    base_game = rsgame.emptygame([3, 2], [2, 3])
    async with mockserver.server() as server:
        game0 = add_singleton_role(base_game, 0, 'a', 'sx', 1)
        sched0 = await get_eosched(server, game0, 'game0')
        game1 = add_singleton_role(base_game, 1, 'r00', 's0', 3)
        sched1 = await get_eosched(server, game1, 'game1')
        with stdout() as out, stderr() as err:
            assert await run('trace', sched0, sched1), err.getvalue()
        traces = json.loads(out.getvalue())
    verify_trace_json(base_game, traces)


@pytest.mark.asyncio
async def test_boot_game(game_info, tmpdir):
    game, sched = game_info
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdout() as out, stderr() as err:
        assert await run(
            'boot', sched, mix_file, '10'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'total', 'r0', 'r1'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()


@pytest.mark.asyncio
async def test_boot_symmetric(tmpdir):
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            'boot', 'game', mix_file, '10'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'surplus', 'regret', 'response'} == results.keys()


@pytest.mark.asyncio
async def test_boot_symmetric_percs(tmpdir):
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join('mix.json'))
    with open(mix_file, 'w') as f:
        json.dump(game.mixture_to_json(game.random_mixture()), f)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            'boot', 'game', mix_file, '10', '-p95'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'95', 'mean'} == results.keys()
    assert {'surplus', 'regret', 'response'} == results['mean'].keys()
    assert {'surplus', 'regret'} == results['95'].keys()


@pytest.mark.asyncio
async def test_boot_game_percs(game_info):
    game, sched = game_info
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert await run(
            'boot', sched, '-', '20', '--percentile', '95',
            '-p99'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'mean', '99', '95'} == results.keys()
    for val in results.values():
        assert {'total', 'r0', 'r1'} == val.keys()
    for val in results['mean'].values():
        assert {'surplus', 'regret', 'response'} == val.keys()
    for val in itertools.chain(results['95'].values(), results['99'].values()):
        assert {'surplus', 'regret'} == val.keys()


@pytest.mark.asyncio
async def test_boot_sim(sim_game_info):
    game, sched, _ = sim_game_info
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert await run(
            'boot', sched, '-', '50', '--chunk-size', '10'), err.getvalue()
        results = json.loads(out.getvalue())
    assert {'total', 'buyers', 'sellers'} == results.keys()
    for val in results.values():
        assert {'surplus', 'regret', 'response'} == val.keys()
