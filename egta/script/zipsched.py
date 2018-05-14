"""Create a zip scheduler from the command line"""
import argparse
import json
import sys

from gameanalysis import gamereader

from egta import zipsched
from egta.script import utils


def add_parser(subparsers):
    """Create zipsched parser"""
    parser = subparsers.add_parser(
        'zip', help="""An egtaonline zip file""", description="""Get payoffs
        from an egtaonline style simulator.  This won't be as efficient as
        using `sim`, but requires no modifications from an existing egtaonline
        scheduler.""")
    parser.add_argument(
        'game', metavar='<game-file>', type=utils.check_file, help="""A file
        with the description of the game to generate profiles from. Only the
        basic game structure is necessary. `-` is interpreted as stdin.""")
    parser.add_argument(
        'zipf', metavar='<zip-file>', type=utils.check_file, help="""The
        zipfile to run. This must be identical to what would be uploaded to
        egtaonline.""")
    parser.add_argument(
        '--conf', metavar='<conf-file>', type=utils.check_file,
        default=argparse.SUPPRESS, help="""A file with the specific
        configuration to load. `-` is interpreted as stdin.  (default: {})""")
    parser.add_argument(
        '--procs', metavar='<num-procs>', type=utils.pos_int,
        default=argparse.SUPPRESS, help="""The maximum number of processes used
        to generate payoff data.""")
    parser.create_scheduler = create_scheduler


async def create_scheduler(game, zipf, procs=4, conf=None, count=1):
    """Create a zip scheduler"""
    max_procs = int(procs)
    count = int(count)

    if game == '-':
        rsgame = gamereader.load(sys.stdin)
    else:
        with open(game) as fil:
            rsgame = gamereader.load(fil)

    if conf is None:
        config = {}
    elif conf == '-':
        config = json.load(sys.stdin)
    else:
        with open(conf) as fil:
            config = json.load(fil)

    return ZipSched(
        rsgame, config, zipf, max_procs=max_procs, simultaneous_obs=count)


class ZipSched(zipsched._ZipScheduler): # pylint: disable=protected-access
    """Zip scheduler that is also async context manager"""
    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return self.__exit__(*args)
