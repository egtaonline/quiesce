"""Script utility for running equilibrium trace"""
import json
import logging

from gameanalysis import merge
from gameanalysis import reduction
from gameanalysis import regret
from gameanalysis import rsgame

import egta.script.eosched as seosched
from egta import countsched
from egta import eosched
from egta import normsched
from egta import savesched
from egta import schedgame
from egta import trace


_log = logging.getLogger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'trace', help="""Compute trace of equilibria between two games""",
        description="""Computes traces of equilibria as the probability of
        mixing between two games changes. This uses quiesce as a subroutine and
        sparsely schedules profiles so that this can be done without having to
        fully sample either game. The results is a list of traces, where each
        trace is a list of the mixture probability, the equilibriium, and the
        regret. In order to allow this to work on games with singleton players,
        games are "normalized" by removing them prior to scheduling. Thus, any
        roles with only one strategy should be omitted from reductions, and
        will also be omitted from equilibria as their answer is trivial. Also
        note that this starts two schedulers so max-schedule is effectively
        doubled.""")
    parser.add_argument(
        'other_id', metavar='<game-id>', type=int, help="""The game id of the
        other game to trace. A mixture value of one corresponds to this game,
        where as a mixture value of zero corresponds to the game and
        configuration initially specified with the scheduler.""")
    parser.add_argument(
        'other_memory', metavar='<other-memory-mb>', type=int, help="""Amount
        of memory in mega bytes to reserve for the other game, specified with
        other-id.""")
    parser.add_argument(
        'other_time', metavar='<other-time-sec>', type=int, help="""Amount of
        time in seconds to give for one simulation of the other game, specified
        with other-id.""")

    parser.add_argument(
        '--regret-thresh', metavar='<reg>', type=float, default=1e-3,
        help="""Regret threshold for a mixture to be considered an equilibrium.
        This can't be strictly enforced as the ODE that this solves has some
        numerical instability, and there is no way to guarantee the equilibria
        stays small.  (default: %(default)g)""")
    parser.add_argument(
        '--dist-thresh', metavar='<norm>', type=float, default=0.1,
        help="""Norm threshold for two mixtures to be considered distinct.
        (default: %(default)g)""")
    parser.add_argument(
        '--max-restrict-size', metavar='<support>', type=int, default=3,
        help="""Support size threshold, beyond which restricted games are not
        required to be explored.  (default: %(default)d)""")
    parser.add_argument(
        '--num-equilibria', metavar='<num>', type=int, default=1,
        help="""Number of equilibria requested to be found. This is mainly
        useful when game contains known degenerate equilibria, but those
        strategies are still useful as deviating strategies. (default:
        %(default)d)""")
    parser.add_argument(
        '--num-backups', metavar='<num>', type=int, default=1, help="""Number
        of backup restricted strategy set to pop at a time, when no equilibria
        are confirmed in initial required set.  When games get to this point
        they can quiesce slowly because this by default pops one at a time.
        Increasing this number can get games like tis to quiesce more quickly,
        but naturally, also schedules more, potentially unnecessary,
        simulations. (default: %(default)d)""")
    parser.add_argument(
        '--dev-by-role', action='store_true', help="""Explore deviations in
        role order instead of all at once. By default, when checking for
        beneficial deviations, all role deviations are scheduled at the same
        time. Setting this will check one role at a time. If a beneficial
        deviation is found, then that restricted strategy set is scheduled
        without exploring deviations from the other roles.""")
    parser.add_argument(
        '--one', action='store_true', help="""Guarantee that an equilibrium is
        found in every restricted game. This may take up to exponential time,
        but a warning will be logged if it takes more than five minutes.""")
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count;role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count;role:count,...>', help="""Specify a
        hierarchical reduction.""")
    return parser


def run(sched, args):
    asched = (sched._sched if isinstance(sched, countsched.CountScheduler)
              else sched)
    asched = (asched._sched if isinstance(asched, savesched.SaveScheduler)
              else asched)
    assert isinstance(asched, seosched.ApiWrapper), \
        "trace currently only supports eo schedulers"
    sched1 = normsched.NormScheduler(sched)

    game = sched1.game()
    if args.dpr is not None:
        red_players = game.role_from_repr(args.dpr, dtype=int)
        red = reduction.deviation_preserving
    elif args.hr is not None:
        red_players = game.role_from_repr(args.hr, dtype=int)
        red = reduction.hierarchical
    else:
        red = reduction.identity
        red_players = None

    egta = asched.api
    egame = egta.get_game(args.other_id).get_observations()
    esim = egta.get_simulator(*egame['simulator_fullname'].split('-', 1))
    with eosched.EgtaOnlineScheduler(
            rsgame.emptygame_json(egame), egta, esim['id'], args.count,
            dict(egame['configuration']), args.sleep, args.max_schedule,
            args.other_memory, args.other_time, args.other_id) as sched2:
        if args.count > 1:
            sched2 = countsched.CountScheduler(sched2, args.count)
        sched2 = normsched.NormScheduler(sched2)

        game1 = schedgame.schedgame(sched1, red, red_players)
        game2 = schedgame.schedgame(sched2, red, red_players)

        traces = trace.trace_equilibria(
            game1, game2, regret_thresh=args.regret_thresh,
            dist_thresh=args.dist_thresh,
            restricted_game_size=args.max_restrict_size,
            num_equilibria=args.num_equilibria, num_backups=args.num_backups,
            devs_by_role=args.dev_by_role, at_least_one=args.one)

        max_reg = 0.0
        spans = [[-1, -1]]
        jtraces = []
        for ts, teqa in traces:
            # compute spans
            span = spans[-1]
            start, *_, end = ts
            if start <= span[1]:
                span[1] = max(span[1], end)
            else:
                spans.append([start, end])

            # serialize trace
            jtrace = []
            for t, eqm in zip(ts, teqa):
                reg = regret.mixture_regret(merge.merge(game1, game2, t), eqm)
                max_reg = max(max_reg, reg)
                jtrace.append({
                    't': float(t),
                    'equilibrium': game.mixture_to_json(eqm),
                    'regret': float(reg)})
            jtraces.append(jtrace)

    _log.error(
        "tracing finished finding %d traces covering %s, with maximum regret "
        "%g", len(jtraces),
        ' U '.join('[{:g}, {:g}]'.format(s, e) for s, e in spans[1:]), max_reg)

    json.dump(jtraces, args.output)
    args.output.write('\n')
