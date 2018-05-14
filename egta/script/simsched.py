"""Creation script for simulation scheduler"""
import argparse
import json
import sys

from gameanalysis import gamereader

from egta import simsched
from egta.script import utils


def add_parser(subparsers):
    """Create simsched parser"""
    parser = subparsers.add_parser(
        'sim', help="""A command line simulator""", description="""Get payoffs
        from a command line simulator.  The simulator will get passed a new
        compressed simulation spec file on each line of stdin, and is expected
        to write a standard observation file to each line of stdout in the same
        order.""")
    parser.add_argument(
        'game', metavar='<game-file>', type=utils.check_file, help="""A file
        with the description of the game to generate profiles from. Only the
        basic game structure is necessary. `-` is interpreted as stdin.""")
    # TODO Ideally this will be nargs, and then could check before joining that
    # there were no spaces.
    parser.add_argument(
        'command', metavar='<command>', help="""The command to run. This is
        space delimited, heavily restricting what can actually be passed.
        Currently the best work around is writing a simple shell wrapper and
        then executing `bash wrapper.sh`.""")
    parser.add_argument(
        '--conf', metavar='<conf-file>', type=utils.check_file,
        default=argparse.SUPPRESS, help="""A file with the specific
        configuration to load. `-` is interpreted as stdin.  (default: {})""")
    parser.add_argument(
        '--buff', metavar='<bytes>', default=argparse.SUPPRESS,
        type=utils.pos_int, help="""Maximum line buffer to prevent deadlock
        with the subprocess.  This default is fine unless you know what you're
        doing.""")
    parser.create_scheduler = create_scheduler


async def create_scheduler(game, command, conf=None, buff=65536, count=1):
    """Create a simulation scheduler"""
    assert int(count) > 0
    buff_size = int(buff)

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

    return simsched.simsched(
        rsgame, config, command.split(), buff_size=buff_size)
