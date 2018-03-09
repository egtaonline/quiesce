import sys

from gameanalysis import paygame
from gameanalysis import gamereader

from egta import gamesched


def create_scheduler(game='-', sample=None, **_):
    if game == '-':
        rsgame = gamereader.load(sys.stdin)
    else:
        with open(game) as f:
            rsgame = gamereader.load(f)

    if sample is not None:
        return Sgs(paygame.samplegame_copy(rsgame))
    else:
        return Rgs(rsgame)


class Sgs(gamesched.SampleGameScheduler):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Rgs(gamesched.RsGameScheduler):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass
