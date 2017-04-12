import numpy as np
from gameanalysis import nash
from gameanalysis import reduction
from gameanalysis import rsgame


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
        (default: %(default)f)""")
    parser.add_argument(
        '--dist-thresh', metavar='<norm>', type=float, default=1e-3,
        help="""Norm threshold for two mixtures to be considered distinct.
        (default: %(default)f)""")
    return parser


def parse_reduction(red):
    strats = (s.strip().split(':') for s in red.strip().split(','))
    return {r.strip(): int(c) for r, c in strats}


def find_eqa(scheduler, game, serial, args):
    if args.dpr is not None:
        red = reduction.DeviationPreserving(
            game.num_strategies, game.num_players,
            serial.from_role_json(parse_reduction(args.dpr), dtype=int))
    else:
        red = reduction.Identity(game.num_strategies, game.num_players)

    profiles = red.expand_profiles(red.red_game.all_profiles())
    promises = [scheduler.schedule(prof) for prof in profiles]
    payoffs = np.concatenate([prom.get()[None] for prom in promises])
    red_game = red.reduce_game(rsgame.game_copy(
        red.full_game, profiles, payoffs))
    eqa = nash.mixed_nash(red_game, regret_thresh=args.regret_thresh,
                          dist_thresh=args.dist_thresh)
    return eqa
