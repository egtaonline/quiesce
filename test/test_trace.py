import numpy as np
import pytest
import timeout_decorator
from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame

from egta import gamesched
from egta import innerloop
from egta import savesched
from egta import schedgame
from egta import trace
from test import utils


games = [
    ([1], [2]),
    ([2], [2]),
    ([2, 1], [1, 2]),
    ([1, 2], [2, 1]),
    ([2, 2], [2, 2]),
    ([3, 2], [2, 3]),
    ([1, 1, 1], [2, 2, 2]),
]


def verify_complete_traces(traces):
    """Verify that traces are in order and complete"""
    t = 0.0
    for (first, *_, last), _ in traces:
        assert first <= t
        t = max(t, last)
    assert t == 1.0


# These sometimes take a really long time because of at_least_one and many
# innerloops. If it takes more than a minute, just give up.
@utils.warnings_filter(RuntimeWarning)
@timeout_decorator.timeout(20)
@pytest.mark.xfail(raises=timeout_decorator.timeout_decorator.TimeoutError)
@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('_', range(5))
def test_random_trace_game(players, strats, _):
    game1 = gamegen.game(players, strats)
    game2 = gamegen.game(players, strats)
    traces = trace.trace_equilibria(game1, game2, at_least_one=True)
    verify_complete_traces(traces)


# These sometimes take a really long time because of at_least_one and many
# innerloops. If it takes more than a minute, just give up.
@utils.warnings_filter(RuntimeWarning)
@timeout_decorator.timeout(20)
@pytest.mark.xfail(raises=timeout_decorator.timeout_decorator.TimeoutError)
@pytest.mark.parametrize('players,strats', games)
@pytest.mark.parametrize('_', range(5))
def test_random_trace_sched(players, strats, _):
    game1 = gamegen.game(players, strats)
    game2 = gamegen.game(players, strats)
    with gamesched.RsGameScheduler(game1) as sched1, \
            gamesched.RsGameScheduler(game2) as sched2:
        traces = trace.trace_equilibria(
            schedgame.schedgame(sched1), schedgame.schedgame(sched2),
            at_least_one=True)
    verify_complete_traces(traces)


def test_sparse_trace():
    """Test that tracing sparsely samples profiles"""
    base = rsgame.emptygame(4, 3)
    game1 = paygame.game_replace(
        base, base.all_profiles(), (base.all_profiles() > 0) * [1, 0, 0])
    game2 = paygame.game_replace(
        base, base.all_profiles(), (base.all_profiles() > 0) * [-1, 2, 0])
    save1 = savesched.SaveScheduler(gamesched.RsGameScheduler(game1))
    save2 = savesched.SaveScheduler(gamesched.RsGameScheduler(game2))
    with save1, save2:
        sgame1 = schedgame.schedgame(save1)
        sgame2 = schedgame.schedgame(save2)

        innerloop.inner_loop(sgame1)
        innerloop.inner_loop(sgame2)

        # Assert that innerloop doesn't scheduler all profiles
        assert save1.get_samplegame().num_profiles == 11
        assert save2.get_samplegame().num_profiles == 11

        ((s1, *_, e1), _), ((s2, *_, e2), _) = trace.trace_equilibria(
            sgame1, sgame2)

    # Assert that trace found the expected equilibria
    assert np.isclose(s1, 0)
    assert np.isclose(e1, 0.25, atol=1e-3)
    assert np.isclose(s2, 0.25, atol=1e-3)
    assert np.isclose(e2, 1)

    # Assert that trace didn't need any extra profiles
    assert save1.get_samplegame().num_profiles == 12
    assert save2.get_samplegame().num_profiles == 12
