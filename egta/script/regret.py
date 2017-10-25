import argparse
import json

import numpy as np
from gameanalysis import rsgame

from egta import regret


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'regret', help="""Compute the regret of a mixture""",
        description="""Samples profiles to compute a sample regret of the
        mixture. By optionally specifying percentiles, bootstrap confidence
        bounds will be returned.""")
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
    means, boots = regret.boot(scheduler, game, mix, args.num,
                               boots=args.boots, chunk_size=args.chunk_size)
    json.dump(dict(zip(('{:g}'.format(p) for p in args.percentiles),
                       np.percentile(boots.max(1), args.percentiles)),
                   mean=means.max()), args.output)
    args.output.write('\n')
