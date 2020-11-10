"""Test tracing"""
import asyncio

import numpy as np
import pytest
from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame

from egta import gamesched
from egta import innerloop
from egta import savesched
from egta import asyncgame
from egta import schedgame
from egta import trace
from test import utils  # pylint: disable=wrong-import-order


def verify_complete_traces(traces):
    """Verify that traces are in order and complete"""
    time = 0.0
    for (first, *_, last), _ in traces:
        assert first <= time
        time = max(time, last)
    assert time == 1.0


# These sometimes take a really long time because of at_least_one and many
# innerloops. If it takes more than a minute, just give up.
@pytest.mark.asyncio
@utils.timeout(20)
@pytest.mark.parametrize("base", utils.games())
async def test_random_trace_game(base):
    """Test tracing for random games"""
    agame1 = asyncgame.wrap(gamegen.game_replace(base))
    agame2 = asyncgame.wrap(gamegen.game_replace(base))
    traces = await trace.trace_all_equilibria(agame1, agame2, style="one")
    verify_complete_traces(traces)


# These sometimes take a really long time because of at_least_one and many
# innerloops. If it takes more than a minute, just give up.
@pytest.mark.asyncio
@utils.timeout(20)
@pytest.mark.parametrize("base", utils.games())
async def test_random_trace_sched(base):
    """Test tracing for random schedulers"""
    sched1 = gamesched.gamesched(gamegen.game_replace(base))
    sched2 = gamesched.gamesched(gamegen.game_replace(base))
    traces = await trace.trace_all_equilibria(
        schedgame.schedgame(sched1), schedgame.schedgame(sched2), style="one"
    )
    verify_complete_traces(traces)


@pytest.mark.asyncio
async def test_sparse_trace():
    """Test that tracing sparsely samples profiles"""
    base = rsgame.empty(4, 3)
    game1 = paygame.game_replace(
        base, base.all_profiles(), (base.all_profiles() > 0) * [1, 0, 0]
    )
    game2 = paygame.game_replace(
        base, base.all_profiles(), (base.all_profiles() > 0) * [-0.5, 1.5, 0]
    )
    save1 = savesched.savesched(gamesched.gamesched(game1))
    save2 = savesched.savesched(gamesched.gamesched(game2))

    sgame1 = schedgame.schedgame(save1)
    sgame2 = schedgame.schedgame(save2)

    await asyncio.gather(innerloop.inner_loop(sgame1), innerloop.inner_loop(sgame2))

    # Assert that innerloop doesn't scheduler all profiles
    assert save1.get_game().num_profiles == 11
    assert save2.get_game().num_profiles == 11

    ((st1, *_, en1), _), (
        (st2, *_, en2),
        _,
    ) = await trace.trace_all_equilibria(  # pylint: disable=too-many-star-expressions
        sgame1, sgame2
    )

    # Assert that trace found the expected equilibria
    assert np.isclose(st1, 0)
    assert np.isclose(en1, 1 / 3, atol=1e-3)
    assert np.isclose(st2, 1 / 3, atol=1e-3)
    assert np.isclose(en2, 1)

    # Assert that trace didn't need many extra profiles
    assert save1.get_game().num_profiles == 12
    assert save2.get_game().num_profiles == 12


@pytest.mark.asyncio
async def test_merge_trace():
    """Test that traces are merged"""
    game0 = asyncgame.wrap(
        paygame.game(2, 2, [[2, 0], [1, 1], [0, 2]], [[0, 0], [1, 1], [0, 0]])
    )
    game1 = asyncgame.wrap(
        paygame.game(2, 2, [[2, 0], [1, 1], [0, 2]], [[0, 0], [1, 1], [0, 3]])
    )
    traces = await trace.trace_all_equilibria(game0, game1)
    assert len(traces) == 1
