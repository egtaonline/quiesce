from egtaonline import api
from gameanalysis import rsgame

from egta import eosched


def create_scheduler(
        game=None, mem=None, time=None, auth=None, count='1', sleep='600',
        max='100', **_):
    assert game is not None, "`game` must be specified"
    assert mem is not None, "`mem` must be specified"
    assert time is not None, "`time` must be specified"
    game_id = int(game)
    mem = int(mem)
    time = int(time)
    count = int(count)
    sleep = float(sleep)
    max_sims = int(max)

    egta = api.EgtaOnlineApi(auth_token=auth)
    with egta:
        game = rsgame.emptygame_json(egta.get_game(game_id).get_summary())

    return ApiWrapper(game, egta, game_id, sleep, count, max_sims, mem, time)


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
