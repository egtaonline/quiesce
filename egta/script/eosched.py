"""Command line creation for egtaonline scheduler"""
from egtaonline import api
from gameanalysis import rsgame
from gameanalysis import utils

from egta import eosched


async def create_scheduler( # pylint: disable=too-many-arguments
        game: """The game id of the game to get profiles from.
        (required)""" = None,
        mem: 'Memory in MB for the scheduler. (required)' = None,
        time: 'Time in seconds for an observation. (required)' = None,
        auth: 'Auth string for egtaonline.' = None,
        count: """This scheduler is more efficient at sampling several
        profiles.""" = '1',
        sleep: """Time to wait in seconds before querying egtaonline to see if
        profiles are complete. Due to the way flux and egtaonline schedule
        jobs, this never needs to be less than 300.""" = '600',
        # pylint: disable-msg=redefined-builtin
        max: """The maximum number of jobs to have active on egtaonline at any
        specific time.""" = '100', **_):
    """A scheduler that gets payoff data from egtaonline."""
    utils.check(game is not None, '`game` must be specified')
    utils.check(mem is not None, '`mem` must be specified')
    utils.check(time is not None, '`time` must be specified')
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
