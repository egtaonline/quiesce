from gameanalysis import paygame

from egta import gamesched


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'game', help="""Schedule simulations with noise added to a game""",
        description="""Sample profiles by adding noise to game data.""")
    parser.add_argument(
        '--sample', '-s', action='store_true', help="""Run the game scheduler
        in sample mode, where first sample payoffs are selected randomly before
        adding noise.""")
    return parser


def create_scheduler(game, args, **_):
    if args.sample:
        return gamesched.SampleGameScheduler(paygame.samplegame_copy(game))
    else:
        return gamesched.RsGameScheduler(game)
