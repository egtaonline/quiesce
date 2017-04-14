import pytest

from gameanalysis import gamegen
from gameanalysis import rsgame

from egta import gamesched
from egta import savesched


@pytest.mark.parametrize("_", range(20))
def test_basic_profile(_):
    sgame = rsgame.samplegame_copy(gamegen.role_symmetric_game([4, 3], [3, 4]))
    profs = sgame.all_profiles()

    sched = savesched.SaveScheduler(
        sgame, gamesched.SampleGameScheduler(sgame))
    with sched:
        proms = [sched.schedule(p) for p in profs]
        for p in proms:
            p.get()
    savegame = sched.get_samplegame()
    assert sgame == savegame
