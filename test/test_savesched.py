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

    sched = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    assert rsgame.emptygame_copy(sgame) == rsgame.emptygame_copy(sched.game())
    async with sched:
        await asyncio.gather(*[sched.sample_payoffs(p) for p in profs[:10]])
    savegame = sched.game()

    sched = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    await asyncio.gather(*[sched.sample_payoffs(p) for p in profs])
    savegame = sched.game()
    assert sgame == savegame
