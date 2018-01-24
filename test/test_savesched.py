import pytest

from gameanalysis import gamegen

from egta import gamesched
from egta import savesched


@pytest.mark.parametrize('_', range(20))
def test_basic_profile(_):
    sgame = gamegen.samplegame([4, 3], [3, 4], 0)
    profs = sgame.all_profiles()

    sched = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    with sched:
        proms = [sched.schedule(p) for p in profs[:10]]
        for p in proms:
            p.get()
    savegame = sched.get_samplegame()
    assert sgame != savegame

    sched = savesched.SaveScheduler(gamesched.SampleGameScheduler(sgame))
    with sched:
        proms = [sched.schedule(p) for p in profs]
        for p in proms:
            p.get()
        for p in proms:
            p.get()
    savegame = sched.get_samplegame()
    assert sgame == savegame
