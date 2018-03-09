from egtaonline import api
from gameanalysis import gamereader
from gameanalysis import rsgame

from egta import eosched


def create_scheduler(
        game=None, mem=None, time=None, auth=None, count='1', sleep='600',
        max='100', **_):
    assert game is not None, "`game_id` must be specified"
    assert mem is not None, "`mem` must be specified"
    assert time is not None, "`time` must be specified"
    mem = int(mem)
    time = int(time)
    count = int(count)
    sleep = float(sleep)
    max_sims = int(max)

    egta = api.EgtaOnlineApi(auth_token=auth)
    with egta:
        summ = egta.get_game(int(game)).get_summary()
        sim_id = egta.get_simulator(
            *summ['simulator_fullname'].split('-', 1))['id']
    game = rsgame.emptygame_copy(gamereader.loadj(summ))
    config = dict(summ.get('configuration', ()) or ())

    return ApiWrapper(
        game, egta, sim_id, count, config, sleep, max_sims, mem, time)


class ApiWrapper(eosched.EgtaOnlineScheduler):
    def __init__(self, game, api, *args, **kwargs):
        super().__init__(game, api, *args, **kwargs)
        self.api = api

    async def __aenter__(self):
        self.api.__enter__()
        return await super().__aenter__()

    async def __aexit__(self, *args):
        await super().__aexit__(*args)
        self.api.__exit__(*args)
