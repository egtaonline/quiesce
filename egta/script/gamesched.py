"""Command line module for game scheduler creation"""
import sys

from gameanalysis import paygame
from gameanalysis import gamereader

from egta import gamesched


async def create_scheduler(
        game: 'A file with the game data. `-` is interpreted as stdin.' = '-',
        sample: """Treat the game as a sample game. This doesn't require a
        value, just a trialing colon.""" = None, **_):
    """A scheduler that samples payoff data from an existing game. If sample is
    specified, payoffs will be a random payoff from each payoff for the
    profile."""
    if game == '-':
        rsgame = gamereader.load(sys.stdin)
    else:
        with open(game) as fil:
            rsgame = gamereader.load(fil)

    if sample is not None: # pylint: disable=no-else-return
        return Sgs(paygame.samplegame_copy(rsgame))
    else:
        return Rgs(rsgame)


class Sgs(gamesched.SampleGameScheduler):
    """Async context manager SampleGameScheduler"""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Rgs(gamesched.RsGameScheduler):
    """Async context manager RsGameScheduler"""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass
