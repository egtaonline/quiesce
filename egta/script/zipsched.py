import json
import sys

from gameanalysis import gamereader

from egta import zipsched


async def create_scheduler(
        game: """A file with the description of the game to generate profiles
        from. Only the basic game structure is necessary. `-` is interpreted as
        stdin."""='-',
        procs: """The maximum number of processes used to generate payoff
        data."""='4',
        zipf: """The zipfile to run. This must be identical to what would be
        uploaded to egtaonline. (required)"""=None,
        conf: """A file with the specific configuration to load. `-` is
        interpreted as stdin. (default: {})"""=None,
        count: """This scheduler is more efficient at sampling several
        profiles."""='1', **_):
    """Get payoffs from an egtaonline style simulator. This won't be as
    efficient as using `sim`, but requires no modifications from an existing
    egtaonline scheduler."""
    assert zipf is not None, '"zipf" must be specified'
    max_procs = int(procs)
    count = int(count)

    if game == '-':
        rsgame = gamereader.load(sys.stdin)
    else:
        with open(game) as f:
            rsgame = gamereader.load(f)

    if conf is None:
        config = {}
    elif conf == '-':
        config = json.load(sys.stdin)
    else:
        with open(conf) as f:
            config = json.load(f)

    return ZipSched(
        rsgame, config, zipf, max_procs=max_procs, simultaneous_obs=count)


class ZipSched(zipsched.ZipScheduler):
    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return self.__exit__(*args)
