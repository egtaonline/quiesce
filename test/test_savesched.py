import asyncio
import pytest

from gameanalysis import gamegen
from gameanalysis import rsgame

from egta import gamesched
from egta import savesched


@pytest.mark.asyncio
@pytest.mark.parametrize('_', range(20))
async def test_basic_profile(_):
    sgame = gamegen.samplegame([4, 3], [3, 4], 0)
    profs = sgame.all_profiles()

    sched = savesched.savesched(gamesched.samplegamesched(sgame))
    assert rsgame.emptygame_copy(sgame) == rsgame.emptygame_copy(sched)
    await asyncio.gather(*[sched.sample_payoffs(p) for p in profs[:10]])
    sched.get_game()

    sched = savesched.savesched(gamesched.samplegamesched(sgame))
    await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    savegame = sched.get_game()
    assert sgame == savegame
    assert sgame == sched.get_game()
