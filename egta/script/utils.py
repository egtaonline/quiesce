import itertools

from gameanalysis import reduction

from egta.script import zipsched
from egta.script import simsched
from egta.script import gamesched
from egta.script import eosched


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
    # FIXME Wrap in save and count
    return base
