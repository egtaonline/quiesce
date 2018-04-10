"""Test save scheduler"""
import asyncio
import pytest

from gameanalysis import gamegen
from gameanalysis import rsgame

from egta import gamesched
from egta import savesched


@pytest.mark.asyncio
@pytest.mark.parametrize('_', range(20))
async def test_basic_profile(_):
    """Test that profiles are saved"""
    sgame = gamegen.samplegame([4, 3], [3, 4], 0)
    profs = sgame.all_profiles()

    basesched = gamesched.samplegamesched(sgame)
    sched = savesched.savesched(basesched)
    assert str(sched) == str(basesched)
    assert rsgame.empty_copy(sgame) == rsgame.empty_copy(sched)
    await asyncio.gather(*[sched.sample_payoffs(p) for p in profs[:10]])
    sched.get_game()

    sched = savesched.savesched(gamesched.samplegamesched(sgame))
    await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    savegame = sched.get_game()
    assert sgame == savegame
    assert sgame == sched.get_game()
