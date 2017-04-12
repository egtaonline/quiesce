import argparse

from gameanalysis import gameio
from gameanalysis import rsgame

from egta import gamesched


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'game', help="""Schedule simulations with noise added to a game""",
        description="""Sample profiles by adding noise to game data.""")

    load_type = parser.add_mutually_exclusive_group(required=True)
    load_type.add_argument(
        '--load-game', dest='loaders', action='store_const',
        const=(lambda g: g.get_summary(), gameio.read_game),
        default=argparse.SUPPRESS, help="""Load game with data instead of just
        format.""")
    load_type.add_argument(
        '--load-samplegame', dest='loaders', action='store_const',
        const=(lambda g: g.get_observations(), gameio.read_samplegame),
        default=argparse.SUPPRESS, help="""Load game with sample data instead
        of just format.""")
    return parser


def create_scheduler(game, serial, args, **_):
    if isinstance(game, rsgame.SampleGame):
        return gamesched.SampleGameScheduler(game)
    else:
        return gamesched.GameScheduler(game)
