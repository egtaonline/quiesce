import argparse
import json
import logging

import numpy as np
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis.reduction import deviation_preserving as dpr
from gameanalysis.reduction import identity as ir


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
    parser.add_argument(
        '--subgame', '-s', metavar='<subgame-file>',
        type=argparse.FileType('r'), help="""Specify an optional subgame to
        sample instead of the whole game. Only deviations from the subgame will
        be scheduled.""")
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
    if args.subgame is None:
        sub = np.ones(game.num_strats, bool)
    else:
        sub = game.from_subgame_json(json.load(args.subgame))

    subg = game.subgame(sub)
    devprofs = red.expand_deviation_profiles(game, sub, red_players)
    subprofs = subgame.translate(
        red.expand_profiles(subg, red.reduce_game(
            subg, red_players).all_profiles()), sub)
    profiles = np.concatenate([subprofs, devprofs])
    promises = [scheduler.schedule(prof) for prof in profiles]
    payoffs = np.concatenate([prom.get()[None] for prom in promises])
    game = red.reduce_game(paygame.game_replace(
        game, profiles, payoffs), red_players)
    eqa = subgame.translate(nash.mixed_nash(
        game.subgame(sub), regret_thresh=args.regret_thresh,
        dist_thresh=args.dist_thresh), sub)
    regrets = [regret.mixture_regret(game, eqm) for eqm in eqa]

    _log.error("brute sampling finished finding %d equilibria:\n%s",
               eqa.shape[0], '\n'.join(
                   '{:d}) {} with regret {:g}'.format(
                       i, game.to_mix_repr(eqm), reg)
                   for i, (eqm, reg) in enumerate(zip(eqa, regrets), 1)))

    json.dump([{'equilibrium': game.to_mix_json(eqm), 'regret': reg}
               for eqm, reg in zip(eqa, regrets)], args.output)
    args.output.write('\n')
