import itertools
import json

from gameanalysis import reduction

from egta import countsched
from egta import savesched
from egta.script import eosched
from egta.script import gamesched
from egta.script import simsched
from egta.script import zipsched


def add_reductions(parser):
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count,role:count,...>', help="""Specify a
        hierarchical reduction.""")


def parse_reduction(game, args):
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


types = {
    'zip': zipsched.create_scheduler,
    'sim': simsched.create_scheduler,
    'game': gamesched.create_scheduler,
    'eo': eosched.create_scheduler,
}


def scheduler(string):
    stype, args, *_ = itertools.chain(string.split(':', 1), [''])
    args = dict(s.split(':', 1) for s in args.split(',') if s)
    base = types[stype](**args)
    if 'save' in args:
        base = SaveWrapper(base, args['save'])
    count = int(args.get('count', '1'))
    if count > 1:
        base = CountWrapper(base, count)
    return base


class SaveWrapper(savesched.SaveScheduler):
    def __init__(self, sched, dest):
        super().__init__(sched)
        self._dest = dest

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        with open(self._dest, 'w') as f:
            json.dump(self.get_game().to_json(), f)
        await self._sched.__aexit__(*args)


class CountWrapper(countsched.CountScheduler):
    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._sched.__aexit__(*args)
