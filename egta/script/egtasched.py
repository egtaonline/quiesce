import argparse
import json
import logging
import sys

from egtaonline import api

from egta import egtasched


_log = logging.getLogger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'egta', help="""Schedule simulations through egta online""",
        description="""Sample profiles from egta online.""")
    parser.add_argument(
        '--conf', '-c', metavar='<conf-json>', type=argparse.FileType('r'),
        help="""The configuration file for the simulator. This only needs to be
        specified if a configuration couldn't be found in the specified
        game.""")
    parser.add_argument(
        '--sim-id', metavar='<sim-id>', type=int, help="""The id of the
        simulator to use. If a game id was specified the simulator id will be
        looked up, but it will take a bit longer.""")
    parser.add_argument(
        '--sleep', '-s', metavar='<sleep-seconds>', default=600, type=int,
        help="""Amount of time to sleep in seconds while waiting for output
        from the simulator. Too long and the script will be paused waiting for
        a response from the simulator, too short and a lof of cpu cycles will
        be wasted checking an empty buffer. (default: %(default)d)""")
    parser.add_argument(
        '--max-schedule', metavar='<observations>', default=100, type=int,
        help="""The maximum number of observations to schedule simultaneously.
        This isn't a strict limit, but it won't be violated by much. (default:
        %(default)s)""")
    # FIXME Since these are required, they should be positional not flagged
    parser.add_argument(
        '--sim-memory', '-m', required=True, metavar='<mb>', type=int,
        help="""Maximum memory needed to run simulation. Standard limit per
        core is 4096, but if your simulation needs less, you should request
        less. Note, that if you set it too low you'll get biased samples for
        simulations that require less memory to run which could be very
        bad.""")
    parser.add_argument(
        '--sim-time', '-t', required=True, metavar='<seconds>', type=int,
        help="""Maximum amount of time needed to run a single simulation.
        Longer times will result in simulations taking longer to schedule, but
        shorter times could result in jobs being canceled from going over time
        which will bias payoff results for shorter running simulations.""")
    return parser


def create_scheduler(game, serial, args, configuration=None, simname=None,
                     **_):
    assert configuration is not None or args.conf is not None, \
        "`conf` must be specified or supplied in the game"
    assert simname is not None or args.sim_id is not None, \
        "`sim-id` must be specified or supplied in the game"
    if args.conf is not None:
        configuration = json.load(args.conf)

    if args.sim_id is None:
        with api.EgtaOnlineApi(auth_token=args.auth_string) as ea:
            args.sim_id = next((
                s['id'] for s in ea.get_simulators()
                if '{}-{}'.format(s['name'], s['version']) == simname), None)
    if args.sim_id is None:  # pragma: no cover
        _log.critical('couldn\'t find simulator with name "{}"'.format(
            simname))
        sys.exit(1)

    return egtasched.EgtaScheduler(
        args.sim_id, game, serial, args.count, configuration, args.sleep,
        args.max_schedule, args.sim_memory, args.sim_time, args.auth_string)
