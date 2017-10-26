import argparse
import json

import numpy as np
from gameanalysis import rsgame

from egta import bootstrap


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'bootstrap', aliases=['boot'], help="""Compute the regret and surplus
        of a mixture""", description="""Samples profiles to compute a sample
        regret and sample surplus of the mixture. By optionally specifying
        percentiles, bootstrap confidence bounds will be returned.""")
    parser.add_argument(
        'mixture', metavar='<mixture-file>', type=argparse.FileType('r'),
        help="""A file with the json formatted mixture to compute the regret
        of.""")
    parser.add_argument(
        'num', metavar='<num-samples>', type=int, help="""The number of samples
        to compute for the regret. One sample for each strategy in the game
        will be taken.""")
    parser.add_argument(
        '--percentiles', '-p', metavar='<percentile>,...', default=(),
        type=lambda s: list(map(float, s.split(','))), help="""A comma
        separated list of confidence percentiles to compute. Specifying
        automatically switches this to use a bootstrap method which takes
        memory proportional to the number of bootstrap samples to run.""")
    parser.add_argument(
        '--boots', '-b', metavar='<num-samples>', type=int, default=101,
        help="""The number of bootstrap samples to take if percentiles is
        specified. More samples will produce less variable confidence bounds at
        the cost of more time. (default: %(default)d)""")
    parser.add_argument(
        '--chunk-size', '-c', metavar='<chunk-size>', type=int, help="""Roughly
        the number of profiles to be scheduled simultaneously. This will be set
        to 10 * boots or 1000 if unspecified. The time to run a simulation
        should roughly correspond to the time to process chunk_size payoffs.
        Keeping this on the order of the number of bootstraps keeps the memory
        requirement constant.""")
    return parser


def run(scheduler, game, args):
    game = rsgame.emptygame_copy(game)
    if not args.percentiles:
        args.boots = 0
    mix = game.from_mix_json(json.load(args.mixture))
    means, boots = bootstrap.deviation_payoffs(
        scheduler, game, mix, args.num, boots=args.boots,
        chunk_size=args.chunk_size)

    exp_means = np.add.reduceat(means * mix, game.role_starts)
    exp_boots = np.add.reduceat(boots * mix, game.role_starts, 1)

    gain_means = means - exp_means.repeat(game.num_role_strats)
    gain_boots = boots - exp_boots.repeat(game.num_role_strats, 1)

    reg_means = np.maximum(np.max(gain_means), 0)
    reg_boots = np.maximum(np.max(gain_boots, 1), 0)

    surp_means = exp_means.dot(game.num_role_players)
    surp_boots = exp_boots.dot(game.num_role_players)

    json.dump({
        'surplus': dict(zip(('{:g}'.format(p) for p in args.percentiles),
                            np.percentile(surp_boots, args.percentiles)),
                        mean=surp_means),
        'regret': dict(zip(('{:g}'.format(p) for p in args.percentiles),
                           np.percentile(reg_boots, args.percentiles)),
                       mean=reg_means),
        'gains': dict(zip(('{:g}'.format(p) for p in args.percentiles),
                          (game.to_payoff_json(g) for g
                           in np.percentile(gain_boots, args.percentiles, 0))),
                      mean=game.to_payoff_json(gain_means)),
    }, args.output)
    args.output.write('\n')
