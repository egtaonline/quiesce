import asyncio
import pytest

import numpy as np
from gameanalysis import gamegen
from gameanalysis import utils

from egta import countsched
from egta import gamesched
from egta import savesched


@pytest.mark.asyncio
async def test_basic_profile():
    sgame = gamegen.samplegame([4, 3], [3, 4])
    profs = utils.unique_axis(sgame.random_profiles(20))

    save = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    async with countsched.CountScheduler(save, 10) as sched:
        paylist = await asyncio.gather(*[
            sched.sample_payoffs(p) for p in profs])
        pays = np.stack(paylist)
    assert np.allclose(pays[profs == 0], 0)

    savegame = save.game()
    assert list(savegame.num_samples) == [10]
