"""Tests for cli"""
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


BASE = path.dirname(path.realpath(__file__))
EGTA = path.join(BASE, "..", "bin", "egta")
SIM_DIR = path.join(BASE, "..", "cdasim")
SIM_GAME_FILE = path.join(SIM_DIR, "small_game.json")


async def run(*args):
    """Run a command line and return if it ran successfully"""
    try:
        await main.amain(*args)
    except SystemExit as ex:
        return not int(str(ex))
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return False
    return True


def stdin(inp):
    """Patch stdin with input"""
    return mock.patch.object(sys, "stdin", io.StringIO(inp))


def stdout():
    """Patch stdout and return stringio"""
    return contextlib.redirect_stdout(io.StringIO())


def stderr():
    """Patch stderr and return stringio"""
    return contextlib.redirect_stderr(io.StringIO())


@pytest.fixture(scope="session", name="jgame")
def fix_jgame():
    """Return json version of sim game"""
    with open(SIM_GAME_FILE) as fil:
        return json.load(fil)


@pytest.fixture(scope="session", name="game")
def fix_game(jgame):
    """Return game"""
    return gamegen.gen_profiles(gamereader.loadj(jgame))


@pytest.fixture(scope="session", name="game_sched")
def fix_game_sched(game, tmpdir_factory):
    """Create game scheduler string"""
    game_file = str(tmpdir_factory.mktemp("games").join("game.json"))
    with open(game_file, "w") as fil:
        json.dump(game.to_json(), fil)
    return "game:game:{}".format(game_file)


@pytest.fixture(scope="session", name="sim_sched")
def fix_sim_sched(jgame, tmpdir_factory):
    """Create sim scheduler string"""
    conf_file = str(tmpdir_factory.mktemp("conf").join("conf.json"))
    with open(conf_file, "w") as fil:
        json.dump(jgame["configuration"], fil)
    sim = " ".join(
        [
            path.join(BASE, "..", "bin", "python"),
            path.join(SIM_DIR, "sim.py"),
            "1",
            "--single",
        ]
    )
    return "sim:command:{},conf:{},game:{}".format(sim, conf_file, SIM_GAME_FILE)


@pytest.fixture(scope="session", name="zip_sched")
def fix_zip_sched():
    """Create zip scheduler string"""
    zip_file = path.join(SIM_DIR, "cdasim.zip")
    return "zip:game:{},zipf:{}".format(SIM_GAME_FILE, zip_file)


def assert_term(sleep1, sleep2, *args):
    """Assert that command handles sigterm"""
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


@pytest.mark.asyncio
async def test_help():
    """Test help works"""
    assert not await run()
    assert not await run("--fail")
    with stderr() as err:
        assert await run("--help"), err.getvalue()
    with stderr() as err:
        assert await run("brute", "--help"), err.getvalue()
    with stderr() as err:
        assert await run("quiesce", "--help"), err.getvalue()
    with stderr() as err:
        assert await run("boot", "--help"), err.getvalue()
    with stderr() as err:
        assert await run("trace", "--help"), err.getvalue()
    with stderr() as err:
        assert await run("spec", "--help"), err.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "red",
    [
        [],
        ["--dpr", "buyers:2,sellers:2"],
        ["--hr", "buyers:2,sellers:2"],
    ],
)
async def test_brute_game(game, game_sched, red):
    """Test brute game"""
    with stdout() as out, stderr() as err:
        assert await run("brute", game_sched + ",count:2", *red), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_brute_game_tag(game, game_sched):
    """Test brute game with tag"""
    with stdout() as out, stderr() as err:
        assert await run(
            "--tag", "test", "brute", game_sched + ",count:2"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_brute_game_restriction(game, game_sched):
    """Test brute game with restriction"""
    rest = json.dumps(game.restriction_to_json(game.random_restriction()))
    with stdin(rest), stdout() as out, stderr() as err:
        assert await run(
            "brute", game_sched + ",count:2", "--restrict", "-"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


def test_brute_game_term(game_sched):
    """Test brute game handles sigterm"""
    assert_term(0.5, 0.5, "brute", game_sched + ",count:2")


@pytest.mark.asyncio
async def test_prof_data(game, game_sched, tmpdir):
    """Test game scheduler"""
    prof_file = str(tmpdir.join("profs.json"))
    with stdout() as out, stderr() as err:
        assert await run(
            "brute", game_sched + ",sample:,save:" + prof_file
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])
    with open(prof_file) as fil:
        prof_game = gamereader.load(fil)
    assert rsgame.empty_copy(game) == rsgame.empty_copy(prof_game)


@pytest.mark.asyncio
async def test_sim(game, sim_sched):
    """Test sim sched"""
    with stdout() as out, stderr() as err:
        assert await run(
            "brute", sim_sched, "--dpr", "buyers:2;sellers:2", "--min-reg"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_sim_in(game, sim_sched):
    """Test sim with stdin"""
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            "brute", sim_sched + ",game:-", "--dpr", "buyers:2;sellers:2", "--min-reg"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_zip(game, zip_sched):
    """Test zip scheduler"""
    with stdout() as out, stderr() as err:
        assert await run(
            "brute", zip_sched, "--dpr", "buyers:2;sellers:2", "--min-reg"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_zip_in(game, zip_sched):
    """Test zip with stdin"""
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run(
            "brute", zip_sched + ",game:-", "--dpr", "buyers:2;sellers:2", "--min-reg"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_zip_conf_in(game, zip_sched):
    """Test zip with configuration in stdin"""
    conf = json.dumps({"max_value": 2})
    with stdin(conf), stdout() as out, stderr() as err:
        assert await run(
            "brute", zip_sched + ",conf:-", "--dpr", "buyers:2;sellers:2", "--min-reg"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_zip_conf_file(game, zip_sched, tmpdir):
    """Test zipsched with conf file"""
    conf_file = str(tmpdir.join("conf.json"))
    with open(conf_file, "w") as fil:
        json.dump({"max_value": 2}, fil)
    with stdout() as out, stderr() as err:
        assert await run(
            "brute",
            zip_sched + ",conf:" + conf_file,
            "--dpr",
            "buyers:2;sellers:2",
            "--min-reg",
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


def test_brute_sim_term(sim_sched):
    """Test brute sim sigterm"""
    assert_term(0.5, 0.5, "brute", sim_sched)


@pytest.mark.asyncio
async def test_sim_delayed_fail(tmpdir):
    """Test delayed simulation fail"""
    game = rsgame.empty(4, 3)
    game_file = str(tmpdir.join("game.json"))
    with open(game_file, "w") as fil:
        json.dump(game.to_json(), fil)
    script = str(tmpdir.join("script.sh"))
    with open(script, "w") as fil:
        fil.write("sleep 1 && false")
    sched = "sim:game:{},command:bash {}".format(game_file, script)
    with stdin(json.dumps({})):
        assert not await run("brute", sched)


@pytest.mark.asyncio
async def test_sim_conf(game, sim_sched, tmpdir):
    """Test sim with configuration"""
    conf_file = str(tmpdir.join("conf.json"))
    with open(conf_file, "w") as fil:
        json.dump(
            {
                "markup": "standard",
                "max_value": 1,
                "arrivals": "simple",
                "market": "call",
            },
            fil,
        )
    with stdout() as out, stderr() as err:
        assert await run(
            "brute", "--dpr", "buyers:2;sellers:2", sim_sched + ",conf:" + conf_file
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_sim_conf_in(game, sim_sched):
    """Test sim with configuration in stdin"""
    conf = json.dumps(
        {
            "markup": "standard",
            "max_value": 1,
            "arrivals": "simple",
            "market": "call",
        }
    )
    with stdin(conf), stdout() as out, stderr() as err:
        assert await run(
            "brute", "--dpr", "buyers:2;sellers:2", sim_sched + ",conf:-"
        ), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


@pytest.mark.asyncio
async def test_innerloop(game, game_sched):
    """Test basic inner loop"""
    with stdout() as out, stderr() as err:
        assert await run("quiesce", game_sched), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


def test_innerloop_game_term(game_sched):
    """Test that inner loop game handles sigterm"""
    assert_term(0.5, 0.5, "quiesce", game_sched)


def test_innerloop_sim_term(sim_sched):
    """Test that inner loop sim handles sigterm"""
    assert_term(0.5, 0.5, "quiesce", sim_sched)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "red",
    [
        [],
        ["--dpr", "buyers:2,sellers:2"],
        ["--hr", "buyers:2,sellers:2"],
    ],
)
async def test_innerloop_red(game, game_sched, red):
    """Test inner loop with reductions"""
    with stdout() as out, stderr() as err:
        assert await run("quiesce", game_sched, *red), err.getvalue()
        for eqm in json.loads(out.getvalue()):
            game.mixture_from_json(eqm["equilibrium"])


async def get_eosched(server, game, name):
    """Get an eo sched"""
    async with api.api() as egta:
        sim = await egta.get_simulator(
            server.create_simulator(name, "1", delay_dist=lambda: random.random() / 100)
        )
        await sim.add_strategies(dict(zip(game.role_names, game.strat_names)))
        eogame = await sim.create_game(name, game.num_players)
        await eogame.add_symgroups(
            list(zip(game.role_names, game.num_role_players, game.strat_names))
        )
        return "eo:game:{:d},mem:2048,time:60,sleep:0.1".format(eogame["id"])


@pytest.mark.asyncio
async def test_brute_egta_game(game):
    """Test brute game with egta sched"""
    async with mockserver.server() as server:
        sched = await get_eosched(server, game, "game")
        with stdout() as out, stderr() as err:
            assert await run(
                "brute", sched, "--dpr", "buyers:2;sellers:2", "--min-reg"
            ), err.getvalue()
            for eqm in json.loads(out.getvalue()):
                game.mixture_from_json(eqm["equilibrium"])


def verify_trace_json(game, traces):
    """Verify traced json"""
    for trace in traces:
        for point in trace:
            assert point.keys() == {"regret", "equilibrium", "t"}
            game.mixture_from_json(point["equilibrium"])
        assert utils.is_sorted(trace, key=lambda t: t["t"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "red",
    [
        [],
        ["--dpr", "buyers:2,sellers:2"],
        ["--hr", "buyers:2,sellers:2"],
    ],
)
async def test_trace(game, sim_sched, tmpdir, red):
    """Test tracing"""
    conf_file = str(tmpdir.join("conf.json"))
    with open(conf_file, "w") as fil:
        json.dump(
            {
                "markup": "standard",
                "max_value": 1,
                "arrivals": "simple",
                "market": "call",
            },
            fil,
        )
    prof_data = str(tmpdir.join("data.json"))
    with stdout() as out, stderr() as err:
        assert await run(
            "trace",
            sim_sched + ",save:" + prof_data,
            sim_sched + ",conf:" + conf_file,
            *red
        ), err.getvalue()
    traces = json.loads(out.getvalue())
    verify_trace_json(game, traces)
    with open(prof_data) as fil:
        data = gamereader.load(fil)
    assert rsgame.empty_copy(game) == rsgame.empty_copy(data)


def add_singleton_role(game, index, role_name, strat_name, num_players):
    """Add a singleton role to a game"""
    role_names = list(game.role_names)
    role_names.insert(index, role_name)
    strat_names = list(game.strat_names)
    strat_names.insert(index, [strat_name])
    role_players = np.insert(game.num_role_players, index, num_players)
    return rsgame.empty_names(role_names, role_players, strat_names)


@pytest.mark.asyncio
async def test_trace_norm():
    """Test tracing"""
    base_game = rsgame.empty([3, 2], [2, 3])
    async with mockserver.server() as server:
        game0 = add_singleton_role(base_game, 0, "a", "sx", 1)
        sched0 = await get_eosched(server, game0, "game0")
        game1 = add_singleton_role(base_game, 1, "r00", "s0", 3)
        sched1 = await get_eosched(server, game1, "game1")
        with stdout() as out, stderr() as err:
            assert await run("trace", sched0, sched1), err.getvalue()
        traces = json.loads(out.getvalue())
    verify_trace_json(base_game, traces)


@pytest.mark.asyncio
async def test_boot_game(game, game_sched, tmpdir):
    """Test bootstrapping a game"""
    mix_file = str(tmpdir.join("mix.json"))
    with open(mix_file, "w") as fil:
        json.dump(game.mixture_to_json(game.random_mixture()), fil)
    with stdout() as out, stderr() as err:
        assert await run("boot", game_sched, mix_file, "10"), err.getvalue()
        results = json.loads(out.getvalue())
    assert {"total", "buyers", "sellers"} == results.keys()
    for val in results.values():
        assert {"surplus", "regret", "response"} == val.keys()


@pytest.mark.asyncio
async def test_boot_symmetric(tmpdir):
    """Test bootstrapping a symmetric game"""
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join("mix.json"))
    with open(mix_file, "w") as fil:
        json.dump(game.mixture_to_json(game.random_mixture()), fil)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run("boot", "game:game:-", mix_file, "10"), err.getvalue()
        results = json.loads(out.getvalue())
    assert {"surplus", "regret", "response"} == results.keys()


@pytest.mark.asyncio
async def test_boot_symmetric_percs(tmpdir):
    """Test bootstrapping a symmetric game with percentiles"""
    game = gamegen.game(4, 3)
    mix_file = str(tmpdir.join("mix.json"))
    with open(mix_file, "w") as fil:
        json.dump(game.mixture_to_json(game.random_mixture()), fil)
    with stdin(json.dumps(game.to_json())), stdout() as out, stderr() as err:
        assert await run("boot", "game:game:-", mix_file, "10", "-p95"), err.getvalue()
        results = json.loads(out.getvalue())
    assert {"95", "mean"} == results.keys()
    assert {"surplus", "regret", "response"} == results["mean"].keys()
    assert {"surplus", "regret"} == results["95"].keys()


@pytest.mark.asyncio
async def test_boot_game_percs(game, game_sched):
    """Test bootstrapping a game with percentiles"""
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert await run(
            "boot", game_sched, "-", "20", "--percentile", "95", "-p99"
        ), err.getvalue()
        results = json.loads(out.getvalue())
    assert {"mean", "99", "95"} == results.keys()
    for val in results.values():
        assert {"total", "buyers", "sellers"} == val.keys()
    for val in results["mean"].values():
        assert {"surplus", "regret", "response"} == val.keys()
    for val in itertools.chain(results["95"].values(), results["99"].values()):
        assert {"surplus", "regret"} == val.keys()


@pytest.mark.asyncio
async def test_boot_sim(game, sim_sched):
    """Test bootstrapping a sim scheduler"""
    mix = json.dumps(game.mixture_to_json(game.random_mixture()))
    with stdin(mix), stdout() as out, stderr() as err:
        assert await run(
            "boot", sim_sched, "-", "50", "--chunk-size", "10"
        ), err.getvalue()
        results = json.loads(out.getvalue())
    assert {"total", "buyers", "sellers"} == results.keys()
    for val in results.values():
        assert {"surplus", "regret", "response"} == val.keys()


@pytest.mark.asyncio
async def test_fix_game_sched(game_sched):
    """Test that game scheduler fixture is correct"""
    _, args_str = game_sched.split(":", 1)
    args = dict(kv.split(":", 1) for kv in args_str.split(","))
    with stdout() as out, stderr() as err:
        assert await run("spec", "game", args["game"]), err.getvalue()
    assert sorted(out.getvalue()[:-1]) == sorted(game_sched)


@pytest.mark.asyncio
async def test_fix_sim_sched(sim_sched):
    """Test that sim scheduler fixture is correct"""
    _, args_str = sim_sched.split(":", 1)
    args = dict(kv.split(":", 1) for kv in args_str.split(","))
    with stdout() as out, stderr() as err:
        assert await run(
            "spec",
            "--count",
            "2",
            "sim",
            args["game"],
            args["command"],
            "--conf",
            args["conf"],
        ), err.getvalue()
    assert sorted(out.getvalue()[:-1]) == sorted(sim_sched + ",count:2")


@pytest.mark.asyncio
async def test_fix_zip_sched(zip_sched):
    """Test that zip scheduler fixture is correct"""
    _, args_str = zip_sched.split(":", 1)
    args = dict(kv.split(":", 1) for kv in args_str.split(","))
    with stdout() as out, stderr() as err:
        assert await run("spec", "zip", args["game"], args["zipf"]), err.getvalue()
    assert sorted(out.getvalue()[:-1]) == sorted(zip_sched)


@pytest.mark.asyncio
async def test_game_sched_file_error():
    """Test that game scheduler fixture is correct"""
    assert not await run("spec", "game", "/this/path/does/not/exist")


@pytest.mark.asyncio
async def test_game_sched_count_error():
    """Test that game scheduler fixture is correct"""
    assert not await run("spec", "--count", "-1", "game", "-")
