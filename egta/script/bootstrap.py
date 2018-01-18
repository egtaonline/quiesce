import argparse
import json
import logging

import numpy as np

from egta import bootstrap


_log = logging.getLogger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'bootstrap', aliases=['boot'], help="""Compute the regret and surplus
        of a mixture""", description="""Samples profiles to compute a sample
        regret and sample surplus of the mixture. By optionally specifying
        percentiles, bootstrap confidence bounds will be returned. The result
        is a json dictionary mapping surplus and regret to either "mean" or a
        string representation of the percentile.""")
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
    parser.add_argument(
        '--standard', action='store_true', help="""Force output to be
        consistent irrespective of if percentiles is specified or the game is
        symmetric.""")
    return parser


def run(scheduler, args):
    game = scheduler.game()
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

    role_ind_reg_means = np.fromiter(map(np.argmax, np.split(
        gain_means, game.role_starts[1:])), int, game.num_roles)
    role_reg_means = gain_means[role_ind_reg_means + game.role_starts]
    ind_reg_means = np.argmax(role_reg_means)
    reg_means = role_reg_means[ind_reg_means]

    role_reg_boots = np.maximum.reduceat(gain_boots, game.role_starts, 1)
    reg_boots = role_reg_boots.max(1)

    role_surp_means = exp_means * game.num_role_players
    surp_means = role_surp_means.sum()
    role_surp_boots = exp_boots * game.num_role_players
    surp_boots = role_surp_boots.sum(1)

    reg_percs = np.percentile(reg_boots, args.percentiles)
    surp_percs = np.percentile(surp_boots, args.percentiles)

    _log.error("bootstrap regret finished with regret %g and surplus %g%s",
               reg_means, surp_means, '' if not args.percentiles else
               ':\nPerc   Regret    Surplus\n----  --------  --------\n' +
               '\n'.join('{: 3g}%  {: 8.4g}  {: 8.4g}'.format(p, r, s)
                         for p, r, s
                         in zip(args.percentiles, reg_percs, surp_percs)))

    # format output
    if game.is_symmetric() and not args.standard:
        result = {'surplus': surp_means,
                  'regret': reg_means,
                  'response': game.strat_names[0][role_ind_reg_means[0]]}

        if args.percentiles:
            result = {'mean': result}
            for p, surp, reg in zip(args.percentiles, surp_percs, reg_percs):
                result['{:g}'.format(p)] = {'surplus': surp, 'regret': reg}
    else:
        mean_dev = '{}: {}'.format(
            game.role_names[ind_reg_means],
            game.strat_names[ind_reg_means][role_ind_reg_means[ind_reg_means]])
        result = {'total': {'surplus': surp_means,
                            'regret': reg_means,
                            'response': mean_dev}}
        for role, strats, surp, reg, dev in zip(
                game.role_names, game.strat_names, role_surp_means,
                role_reg_means, role_ind_reg_means):
            result[role] = {'surplus': surp,
                            'regret': reg,
                            'response': strats[dev]}

        if args.percentiles or args.standard:
            result = {'mean': result}

        for p, surp, reg, role_surps, role_regs in zip(
                args.percentiles, surp_percs, reg_percs,
                np.percentile(role_surp_boots, args.percentiles, 0),
                np.percentile(role_reg_boots, args.percentiles, 0)):
            perc = {'total': {'surplus': surp,
                              'regret': reg}}
            for role, surp, reg in zip(game.role_names, role_surps, role_regs):
                perc[role] = {'surplus': surp, 'regret': reg}
            result['{:g}'.format(p)] = perc

    json.dump(result, args.output)
    args.output.write('\n')
