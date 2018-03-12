import argparse
import json
import logging

import numpy as np

from egta import bootstrap
from egta.script import utils


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'bootstrap', aliases=['boot'], help="""Compute the regret and surplus
        of a mixture""", description="""Samples profiles to compute a sample
        regret and sample surplus of the mixture. By optionally specifying
        percentiles, bootstrap confidence bounds will be returned. The result
        is a json dictionary mapping surplus and regret to either "mean" or a
        string representation of the percentile.""")
    parser.add_argument(
        'scheduler', metavar='<sched-spec>', help="""A scheduler specification,
        see below.""")
    parser.add_argument(
        'mixture', metavar='<mixture-file>', type=argparse.FileType('r'),
        help="""A file with the json formatted mixture to compute the regret
        of.""")
    parser.add_argument(
        'num', metavar='<num-samples>', type=int, help="""The number of samples
        to compute for the regret. One sample for each strategy in the game
        will be taken.""")
    parser.add_argument(
        '--percentile', '-p', metavar='<percentile>', default=[],
        action='append', type=float, help="""A confidence percentile to
        compute.  Specifying at least once switches this to use a bootstrap
        method which takes memory proportional to the number of bootstrap
        samples to run.""")
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
    utils.add_scheduler_epilog(parser)
    parser.run = run
    return parser


async def run(args):
    sched = await utils.parse_scheduler(args.scheduler)
    if not args.percentile:
        args.boots = 0
    mix = sched.mixture_from_json(json.load(args.mixture))
    async with sched:
        means, boots = await bootstrap.deviation_payoffs(
            sched, mix, args.num, boots=args.boots, chunk_size=args.chunk_size)

    exp_means = np.add.reduceat(means * mix, sched.role_starts)
    exp_boots = np.add.reduceat(boots * mix, sched.role_starts, 1)

    gain_means = means - exp_means.repeat(sched.num_role_strats)
    gain_boots = boots - exp_boots.repeat(sched.num_role_strats, 1)

    role_ind_reg_means = np.fromiter(map(np.argmax, np.split(
        gain_means, sched.role_starts[1:])), int, sched.num_roles)
    role_reg_means = gain_means[role_ind_reg_means + sched.role_starts]
    ind_reg_means = np.argmax(role_reg_means)
    reg_means = role_reg_means[ind_reg_means]

    role_reg_boots = np.maximum.reduceat(gain_boots, sched.role_starts, 1)
    reg_boots = role_reg_boots.max(1)

    role_surp_means = exp_means * sched.num_role_players
    surp_means = role_surp_means.sum()
    role_surp_boots = exp_boots * sched.num_role_players
    surp_boots = role_surp_boots.sum(1)

    reg_percs = np.percentile(reg_boots, args.percentile)
    surp_percs = np.percentile(surp_boots, args.percentile)

    logging.error("bootstrap regret finished with regret %g and surplus %g%s",
                  reg_means, surp_means, '' if not args.percentile else
                  ':\nPerc   Regret    Surplus\n----  --------  --------\n' +
                  '\n'.join('{: 3g}%  {: 8.4g}  {: 8.4g}'.format(p, r, s)
                            for p, r, s
                            in zip(args.percentile, reg_percs, surp_percs)))

    # format output
    if sched.is_symmetric() and not args.standard:
        result = {'surplus': surp_means,
                  'regret': reg_means,
                  'response': sched.strat_names[0][role_ind_reg_means[0]]}

        if args.percentile:
            result = {'mean': result}
            for p, surp, reg in zip(args.percentile, surp_percs, reg_percs):
                result['{:g}'.format(p)] = {'surplus': surp, 'regret': reg}
    else:
        mean_dev = '{}: {}'.format(
            sched.role_names[ind_reg_means],
            sched.strat_names[ind_reg_means][role_ind_reg_means[
                ind_reg_means]])
        result = {'total': {'surplus': surp_means,
                            'regret': reg_means,
                            'response': mean_dev}}
        for role, strats, surp, reg, dev in zip(
                sched.role_names, sched.strat_names, role_surp_means,
                role_reg_means, role_ind_reg_means):
            result[role] = {'surplus': surp,
                            'regret': reg,
                            'response': strats[dev]}

        if args.percentile or args.standard:
            result = {'mean': result}

        for p, surp, reg, role_surps, role_regs in zip(
                args.percentile, surp_percs, reg_percs,
                np.percentile(role_surp_boots, args.percentile, 0),
                np.percentile(role_reg_boots, args.percentile, 0)):
            perc = {'total': {'surplus': surp,
                              'regret': reg}}
            for role, surp, reg in zip(sched.role_names, role_surps,
                                       role_regs):
                perc[role] = {'surplus': surp, 'regret': reg}
            result['{:g}'.format(p)] = perc

    json.dump(result, args.output)
    args.output.write('\n')
