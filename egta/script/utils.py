"""Utilities for command line modules"""
import argparse
import inspect
import itertools
import json
import shutil
import textwrap
from os import path

from gameanalysis import reduction

from egta import countsched
from egta import savesched
from egta.script import eosched
from egta.script import gamesched
from egta.script import simsched
from egta.script import zipsched


def add_reductions(parser):
    """Add reduction options to a parser"""
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count,role:count,...>', help="""Specify a
        hierarchical reduction.""")


def parse_reduction(game, args):
    """Parse a reduction string"""
    if args.dpr is not None:
        red_players = game.role_from_repr(args.dpr, dtype=int)
        red = reduction.deviation_preserving
    elif args.hr is not None:
        red_players = game.role_from_repr(args.hr, dtype=int)
        red = reduction.hierarchical
    else:
        red_players = None
        red = reduction.identity
    return red, red_players


def pos_int(string):
    """Type for a positive integer"""
    val = int(string)
    if val <= 0:
        raise argparse.ArgumentTypeError(
            '{:d} is an invalid positive int value'.format(string))
    return val


def check_file(string):
    """Type for a file that exists"""
    if not string == '-' and not os.path.isfile(string):
        raise argparse.ArgumentTypeError(
            '{} is not a file'.format(string))
    return string


_TYPES = {
    'zip': zipsched.create_scheduler,
    'sim': simsched.create_scheduler,
    'game': gamesched.create_scheduler,
    'eo': eosched.create_scheduler,
}


async def parse_scheduler(string):
    """Return a scheduler for a string specification"""
    stype, args, *_ = itertools.chain(string.split(':', 1), [''])
    args = dict(s.split(':', 1) for s in args.split(',') if s)
    base = await _TYPES[stype](**args)
    if 'save' in args:
        base = SaveWrapper(base, args['save'])
    count = int(args.get('count', '1'))
    if count > 1:
        base = CountWrapper(base, count)
    return base


class SaveWrapper(savesched._SaveScheduler): # pylint: disable=protected-access
    """Make save scheduler an async context manager"""
    def __init__(self, sched, dest):
        super().__init__(sched)
        self._dest = dest

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        with open(self._dest, 'w') as fil:
            json.dump(self.get_game().to_json(), fil)
        await self._sched.__aexit__(*args)


class CountWrapper(countsched._CountScheduler): # pylint: disable=protected-access
    """Make count scheduler an async context manager"""
    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._sched.__aexit__(*args)
