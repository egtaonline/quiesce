import json

import numpy as np
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis.reduction import deviation_preserving as dpr
from gameanalysis.reduction import identity as ir


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'brute', help="""Compute equilibrium by sampling all profiles""",
        description="""Samples profiles from the entire game, and then runs
        standard equilibrium finding. For games with a large number of players,
        a reduction should be specified.""")
    parser.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a DPR
        reduction.""")
    parser.add_argument(
        '--regret-thresh', metavar='<reg>', type=float, default=1e-3,
        help="""Regret threshold for a mixture to be considered an equilibrium.
        (default: %(default)g)""")
    parser.add_argument(
        '--dist-thresh', metavar='<norm>', type=float, default=1e-3,
        help="""Norm threshold for two mixtures to be considered distinct.
        (default: %(default)g)""")
    return parser


def parse_reduction(red):
    strats = (s.strip().split(':') for s in red.strip().split(','))
    return {r.strip(): int(c) for r, c in strats}


def run(scheduler, game, args):
    game = rsgame.emptygame_copy(game)
    red = ir
    red_players = None
    if args.dpr is not None:
        red_players = game.from_role_json(parse_reduction(args.dpr), dtype=int)
        red = dpr
    redgame = red.reduce_game(game, red_players)
    profiles = red.expand_profiles(game, redgame.all_profiles())
    promises = [scheduler.schedule(prof) for prof in profiles]
    payoffs = np.concatenate([prom.get()[None] for prom in promises])
    redgame = red.reduce_game(paygame.game_replace(
        game, profiles, payoffs), red_players)
    eqa = nash.mixed_nash(redgame, regret_thresh=args.regret_thresh,
                          dist_thresh=args.dist_thresh)
    for eqm in eqa:
        json.dump(game.to_mix_json(eqm), args.output)
        args.output.write('\n')
