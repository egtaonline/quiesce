import argparse
import json

from egta import simsched


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'sim', help="""Schedule simulations with a local simulator""",
        description="""Sample profiles from a local simulator specified as an
        executable.""")
    parser.add_argument(
        '--conf', '-c', metavar='<conf-json>', type=argparse.FileType('r'),
        help="""The configuration file for the simulator. This only needs to be
        specified if a configuration couldn't be found in the specified
        game.""")
    parser.add_argument(
        '--sleep', '-s', metavar='<sleep-seconds>', default=1, type=int,
        help="""Amount of time to sleep in seconds while waiting for output
        from the simulator. Too long and the script will be paused waiting for
        a response from the simulator, too short and a lof of cpu cycles will
        be wasted checking an empty buffer. (default: %(default)d)""")
    parser.add_argument(
        'command', nargs='+', metavar='<command>', help="""Command to
        execute. It must read from stdin where each line is a simulation spec
        file, and print results to stdout. stdout must be flushed after all
        inputs have been processes, otherwise this will hang waiting for a
        flushed buffer. The supplied command will be executed in the current
        directory.""")
    return parser


def create_scheduler(game, serial, args, configuration=None, **_):
    assert configuration is not None or args.conf is not None, \
        "`conf` must be specified or supplied in the game"
    if args.conf is not None:
        configuration = json.load(args.conf)
    return simsched.SimulationScheduler(serial, configuration, args.command,
                                        sleep=args.sleep)
