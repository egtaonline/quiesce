import argparse
import json
import logging

import numpy as np
from gameanalysis import nash
from gameanalysis import reduction
from gameanalysis import regret
from gameanalysis import restrict

from egta import schedgame


_log = logging.getLogger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'brute', help="""Compute equilibria by sampling all profiles""",
        description="""Samples profiles from the entire game, and then runs
        standard equilibrium finding. For games with a large number of players,
        a reduction should be specified. A list of is returned where each
        element has an "equilibrium" and the corresponding "regret" in the full
        game.""")
    parser.add_argument(
        '--regret-thresh', metavar='<reg>', type=float, default=1e-3,
        help="""Regret threshold for a mixture to be considered an equilibrium.
        (default: %(default)g)""")
    parser.add_argument(
        '--dist-thresh', metavar='<norm>', type=float, default=0.1,
        help="""Norm threshold for two mixtures to be considered distinct.
        (default: %(default)g)""")
    parser.add_argument(
        '--restrict', '-r', metavar='<restriction-file>',
        type=argparse.FileType('r'), help="""Specify an optional restricted
        game to sample instead of the whole game. Only deviations from the
        restricted strategy set will be scheduled.""")
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count,role:count,...>', help="""Specify a
        hierarchical reduction.""")
    return parser


def run(scheduler, args):
    game = scheduler.game()
    if args.dpr is not None:
        red_players = game.role_from_repr(args.dpr, dtype=int)
        red = reduction.deviation_preserving
    elif args.hr is not None:
        red_players = game.role_from_repr(args.hr, dtype=int)
        red = reduction.hierarchical
    else:
        red = reduction.identity
        red_players = None

    rest = (np.ones(game.num_strats, bool) if args.restrict is None
            else game.restriction_from_json(json.load(args.restrict)))
    game = schedgame.schedgame(scheduler, red, red_players)
    rgame = game.restrict(rest)

    # schedule all deviations
    game.deviation_payoffs(restrict.translate(rgame.uniform_mixture(), rest))

    # now find equilibria
    eqa = restrict.translate(nash.mixed_nash(
        rgame, regret_thresh=args.regret_thresh,
        dist_thresh=args.dist_thresh), rest)
    reg_info = []
    for eqm in eqa:
        gains = regret.mixture_deviation_gains(game, eqm)
        bri = np.argmax(gains)
        reg_info.append((gains[bri],) + game.role_strat_names[bri])

    _log.error(
        "brute sampling finished finding %d equilibria:\n%s",
        eqa.shape[0], '\n'.join(
            '{:d}) {} with regret {:g} to {} {}'.format(
                i, game.mixture_to_repr(eqm), reg, role, strat)
            for i, (eqm, (reg, role, strat))
            in enumerate(zip(eqa, reg_info), 1)))

    json.dump([{'equilibrium': game.mixture_to_json(eqm),
                'regret': reg,
                'best_response': {'role': role, 'strat': strat}}
               for eqm, (reg, role, strat)
               in zip(eqa, reg_info)], args.output)
    args.output.write('\n')
