import asyncio
import contextlib
import logging

from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils as gu

from egta import profsched
from egta import utils as eu


class EgtaOnlineScheduler(profsched.Scheduler):
    """A profile scheduler that schedules through egta online

    Parameters
    ----------
    game : RsGame
        The gameanalysis basegame representing the game to schedule.
    api : EgtaOnlineApi
        The api object to be uased to query EGTA Online.
    game_id : int
        The id of the egtaonline game to use. It must match the setup of game.
        Use the egtaonline `get_or_create_game` if you need to get a game_id
        from a configuration.
    sleep_time : int
        Time in seconds between queries to egtaonline to determine if profiles
        have finished. This should probably be set roughly equal to the time it
        takes for a simulation to run.
    simultaneous_obs : int
        The number of simultaneous observations to schedule at a time. EGTA
        Online will use this when scheduling.
    max_scheduled : int
        The maximum number of observations to schedule simultaneously. Keeping
        this low helps prevent starving others of flux cycles.
    obs_memory : int
        The amount of memory in MB to allocate for each simulation.
    obs_time : int
        The amount of time in seconds to give each simulation to run. Too low
        and long running simulations will get cancelled giving you biased
        samples, too long and it will take longer to schedule jobs on flux.
    """

    def __init__(self, game, api, game_id, sleep_time, simultaneous_obs,
                 max_scheduled, obs_memory, obs_time):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._api = api
        self._game = paygame.samplegame_copy(rsgame.emptygame_copy(game))
        self._game_id = game_id

        self._sleep_time = sleep_time
        self._obs_memory = obs_memory
        self._obs_time = obs_time
        self._simult_obs = simultaneous_obs

        self._is_open = False
        self._profiles = {}
        self._prof_ids = {}
        self._sched = None
        self._fetcher = None
        self._sched_lock = asyncio.Lock()
        self._scheduled = asyncio.BoundedSemaphore(
            max_scheduled * simultaneous_obs)

    def _check_fetcher(self):
        if self._fetcher.done() and self._fetcher.exception() is not None:
            raise self._fetcher.exception()

    async def sample_payoffs(self, profile):
        gu.check(self._is_open, 'not open')
        self._check_fetcher()
        hprof = gu.hash_array(profile)
        data = self._profiles.setdefault(
            hprof, ([0], [0], [0], [None], asyncio.Queue()))
        scheduled, _, claimed, prof_id, pays = data
        claimed[0] += 1
        if scheduled[0] < claimed[0]:
            # TODO make requests async
            scheduled[0] += self._simult_obs
            async with self._sched_lock:
                for _ in range(self._simult_obs):
                    await self._scheduled.acquire()
                pid = prof_id[0]
                if pid is not None:
                    await self._sched.remove_profile(pid)
                assignment = self._game.profile_to_repr(profile)
                prof_id[0] = (await self._sched.add_profile(
                    assignment, scheduled[0]))['id']
                if pid is None:
                    self._prof_ids[prof_id[0]] = data
        pay = await pays.get()
        self._check_fetcher()
        return pay

    async def _fetch(self):
        # TODO Make requests async
        try:
            while True:
                logging.info('query scheduler %d for game %d',
                             self._sched['id'], self._game_id)
                info = await self._sched.get_requirements()
                gu.check(info['active'], 'scheduler was deactivated')
                reqs = info['scheduling_requirements']
                for req in reqs:
                    prof_id = req['id']
                    scheduled, received, _, _, pays = self._prof_ids[prof_id]
                    if req['current_count'] <= received[0]:
                        continue
                    egta_prof = await self._api.get_profile(prof_id)
                    jobs = await egta_prof.get_observations()
                    # TODO Is this still necessary
                    # valid = all(o['symmetry_groups'] is not None for o
                    #             in jobs['observations'])
                    obs = self._game.samplepay_from_json(jobs)
                    num = obs.shape[0] - received[0]
                    # Only un-schedule the amount different than what you
                    # thought you scheduled. We can get other observations from
                    # other schedulers.
                    for _ in range(min(num, scheduled[0] - received[0])):
                        self._scheduled.release()
                    received[0] += num
                    obs = obs[:num].copy()
                    obs.setflags(write=False)
                    for o in obs:
                        pays.put_nowait(o)
                await asyncio.sleep(self._sleep_time)
        except Exception as ex:
            for _, (received,), (claimed,), _, pays in self._profiles.values():
                for _ in range(claimed - received):
                    pays.put_nowait(None)
            raise ex

    async def open(self):
        gu.check(not self._is_open, 'already open')
        try:
            game = await self._api.get_game(self._game_id)
            obs = await game.get_observations()
            gu.check(
                rsgame.emptygame_copy(self._game) ==
                rsgame.emptygame_json(obs),
                "egtaonline game didn't match specified game")
            conf = dict(obs.get('configuration', ()) or ())
            profiles = obs.get('profiles', ()) or ()

            # Parse profiles
            num_profs = len(profiles)
            num_pays = 0
            for jprof in profiles:
                pid = jprof['id']
                prof, spays = self._game.profsamplepay_from_json(jprof)
                spays.setflags(write=False)
                hprof = gu.hash_array(prof)
                pays = asyncio.Queue()
                num_spays = len(spays)
                num_pays += num_spays
                for pay in spays:
                    pays.put_nowait(pay)
                data = ([num_spays], [num_spays], [0], [pid], pays)
                self._profiles[hprof] = data
                self._prof_ids[pid] = data
            logging.info(
                'found %d existing profiles with %d payoffs in game %d',
                num_profs, num_pays, self._game_id)

            # Create and start scheduler
            self._sched = await obs.create_generic_scheduler(
                'egta_' + eu.random_string(20), True, self._obs_memory,
                self._obs_time, self._simult_obs, 1, conf)
            logging.warning(
                'created scheduler %d for running simulations of game %d: '
                'https://%s/generic_schedulers/%d', self._sched['id'],
                self._game_id, self._api.domain, self._sched['id'])
            self._fetcher = asyncio.ensure_future(self._fetch())
            self._is_open = True
        except Exception as ex:
            await self.close()
            raise ex
        return self

    async def close(self):
        if self._fetcher is not None:
            self._fetcher.cancel()
            with contextlib.suppress(Exception):
                await self._fetcher
            self._fetcher = None

        if self._sched is not None:
            await self._sched.deactivate()
            logging.info('deactivated scheduler %d for game %d',
                         self._sched['id'], self._game_id)
            self._sched = None

        if self._sched_lock.locked():
            self._sched_lock.release()
        while True:
            try:
                self._scheduled.release()
            except ValueError:
                break  # Fully reset
        self._profiles.clear()
        self._prof_ids.clear()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __str__(self):
        return str(self._game_id)


def eosched(
        game, api, game_id, sleep_time, simultaneous_obs, max_scheduled,
        obs_memory, obs_time):
    return EgtaOnlineScheduler(
        game, api, game_id, sleep_time, simultaneous_obs, max_scheduled,
        obs_memory, obs_time)
