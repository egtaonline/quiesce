"""Command line creation for egtaonline scheduler"""
import argparse

from egtaonline import api
from gameanalysis import rsgame

from egta import eosched
from egta.script import utils


def add_parser(subparsers):
    """Create eosched parser"""
    parser = subparsers.add_parser(
        'eo', help="""Egtaonline""", description="""A scheduler that gets
        payoff data from egtaonline.""")
    parser.add_argument(
        'game', metavar='<game-id>', type=utils.pos_int, help="""The game id of
        the game to get profiles from.""")
    parser.add_argument(
        'mem', metavar='<mega-bytes>', type=utils.pos_int, help="""Memory in MB
        for the scheduler.""")
    parser.add_argument(
        'time', metavar='<seconds>', type=utils.pos_int, help="""Time in
        seconds for an observation.""")
    parser.add_argument(
        '--auth', metavar='<auth-token>', default=argparse.SUPPRESS,
        help="""Auth string for egtaonline.""")
    parser.add_argument(
        '--sleep', default=argparse.SUPPRESS, metavar='<seconds>',
        type=utils.pos_int, help="""Time to wait in seconds before querying
        egtaonline to see if profiles are complete.  Due to the way flux and
        egtaonline schedule jobs, this never needs to be less than 300.""")
    parser.add_argument(
        '--max', default=argparse.SUPPRESS, metavar='<num-jobs>',
        type=utils.pos_int, help="""The maximum number of jobs to have active
        on egtaonline at any specific time.""")
    parser.create_scheduler = create_scheduler


async def create_scheduler( # pylint: disable=too-many-arguments
        game, mem, time, auth=None, count=1, sleep=600,
        # pylint: disable-msg=redefined-builtin
        max=100):
    """Create an egtaonline scheduler"""
    game_id = int(game)
    mem = int(mem)
    time = int(time)
    count = int(count)
    sleep = float(sleep)
    max_sims = int(max)

    async with api.api(auth_token=auth) as egta:
        game = await egta.get_game(game_id)
        summ = await game.get_summary()
    game = rsgame.empty_json(summ)
    return ApiWrapper(game, egta, game_id, sleep, count, max_sims, mem, time)


class ApiWrapper(eosched._EgtaOnlineScheduler): # pylint: disable=protected-access
    """Wrapper for egtaonline api to open on open of scheduler"""
    def __init__(self, game, eoapi, *args, **kwargs):
        super().__init__(game, eoapi, *args, **kwargs)
        self.api = eoapi

    async def __aenter__(self):
        await self.api.__aenter__()
        return await super().__aenter__()

    async def __aexit__(self, *args):
        await super().__aexit__(*args)
        await self.api.__aexit__(*args)
