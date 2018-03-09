import json
import sys

from gameanalysis import gamereader

from egta import zipsched


def create_scheduler(
        game='-', procs='4', zipf=None, conf=None, count='1', **_):
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
