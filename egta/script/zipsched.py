import argparse
import io
import json

from egta import zipsched


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'zip', help="""Schedule simulations with a local zip simulator""",
        description="""Sample profiles from a local simulator specified as an
        EGTA Online zipfile.""")
    parser.add_argument(
        '--conf', '-c', metavar='<conf-json>', default=io.StringIO('{}'),
        type=argparse.FileType('r'), help="""The configuration file for the
        simulator. This only needs to be specified to add configuration
        parameters. The defaults.json will be used in addition.""")
    parser.add_argument(
        '--max-procs', '-p', metavar='<max-procs>', default=4, type=int,
        help="""The maximum number of independent processes to spawn for zip
        file output. Ideally this will be about one more than the number of
        processes on your machine. (default: %(default)s)""")
    parser.add_argument(
        'zip', metavar='<zip-file>', help="""EGTA Online zip file to execute.
        See EGTA Online for the proper format.""")
    return parser


def create_scheduler(game, args, configuration={}, **_):
    conf = configuration.copy()
    conf.update(json.load(args.conf))
    return zipsched.ZipScheduler(game, configuration, args.zip,
                                 max_procs=args.max_procs)
