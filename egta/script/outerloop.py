import argparse
import json

from gameanalysis import reduction

from egta import outerloop


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'quiesce', help="""Compute equilibrium using the quiesce procedure""",
        description="""Samples profiles from small subgames, expanding subgame
        support by best responses to candidate subgame equilibria. For games
        with a large number of players, a reduction should be specified.""")
    parser.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a DPR
        reduction.""")
    parser.add_argument(
        '--initial-subgame', metavar='<json-file>',
        type=argparse.FileType('r'), help="""Specify an initial subgame to
        outerloop over the remaining strategies in the game. If this is
        specified the first line in output will be the final subgame after
        outerlooping, and the remaining lines will be the equilibria.""")

    parser.add_argument(
        '--regret-thresh', metavar='<reg>', type=float, default=1e-3,
        help="""Regret threshold for a mixture to be considered an equilibrium.
        (default: %(default)s)""")
    parser.add_argument(
        '--dist-thresh', metavar='<norm>', type=float, default=1e-3,
        help="""Norm threshold for two mixtures to be considered distinct.
        (default: %(default)s)""")
    parser.add_argument(
        '--max-resamples', metavar='<resamples>', type=int, default=10,
        help="""Number of times to resample all profiles in a subgame when
        equilibrium is not identified.  (default: %(default)d)""")
    parser.add_argument(
        '--max-subgame-size', metavar='<support>', type=int, default=3,
        help="""Support size threshold, beyond which subgames are not required
        to be explored.  (default: %(default)d)""")
    parser.add_argument(
        '--num-equilibria', metavar='<num>', type=int, default=1,
        help="""Number of equilibria requested to be found. This is mainly
        useful when game contains known degenerate equilibria, but those
        strategies are still useful as deviating strategies. (default:
        %(default)d)""")
    parser.add_argument(
        '--num-backups', metavar='<num>', type=int, default=1, help="""Number
        of backup subgames to pop at a time, when no equilibria are confirmed
        in initial required set.  When games get to this point they can quiesce
        slowly because this by default pops one at a time. Increasing this
        number can get games like tis to quiesce more quickly, but naturally,
        also schedules more, potentially unnecessary, simulations. (default:
        %(default)d)""")
    parser.add_argument(
        '--dev-by-role', action='store_true', help="""Explore deviations in
        role order instead of all at once. By default, when checking for
        beneficial deviations, all role deviations are scheduled at the same
        time. Setting this will check one role at a time. If a beneficial
        deviation is found, then that subgame is scheduled without exploring
        deviations from the other roles.""")
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

    initial_subgame = args.initial_subgame
    if initial_subgame is not None:
        initial_subgame = serial.from_subgame_json(
            json.load(initial_subgame))

    eqa, subgame = outerloop.outer_loop(
        scheduler, game, initial_subgame, red=red,
        regret_thresh=args.regret_thresh, dist_thresh=args.dist_thresh,
        max_resamples=args.max_resamples, subgame_size=args.max_subgame_size,
        num_equilibria=args.num_equilibria, num_backups=args.num_backups,
        devs_by_role=args.dev_by_role)

    if initial_subgame is not None:
        json.dump(serial.to_subgame_json(subgame), args.output)
        args.output.write('\n')
    return eqa
