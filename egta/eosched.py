import logging
import queue
import threading

import numpy as np
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils as gu

from egta import profsched
from egta import utils as eu


_log = logging.getLogger(__name__)


class EgtaOnlineScheduler(profsched.Scheduler):
    """A profile scheduler that schedules through egta online

    Parameters
    ----------
    game : RsGame
        The gameanalysis basegame representing the game to schedule.
    api : EgtaOnlineApi
        The api object to be uased to query EGTA Online.
    sim_id : int
        The id of the egtaonline simulator to use.
    simultaneous_obs : int
        The number of simultaneous observations to schedule at a time. EGTA
        Online will use this when scheduling.
    configuration : {key: value}
        Dictionary of configuration to use with the scheduler. Any fields that
        are omitted will be filled in by the defaults specified in the
        simulator.
    sleep_time : int
        Time in seconds between queries to egtaonline to determine if profiles
        have finished. This should probably be set roughly equal to the time it
        takes for a simulation to run.
    max_scheduled : int
        The maximum number of observations to schedule simultaneously. Keeping
        this low helps prevent starving others of flux cycles.
    obs_memory : int
        The amount of memory in MB to allocate for each simulation.
    obs_time : int
        The amount of time in seconds to give each simulation to run. Too low
        and long running simulations will get cancelled giving you biased
        samples, too long and it will take longer to schedule jobs on flux.
    game_id : int, optional
        An optional game id corresponding to a game that matches the
        configuration. This prevents creating a temporary game just to query
        initial profile data.
    """

    def __init__(self, game, api, sim_id, simultaneous_obs, configuration,
                 sleep_time, max_scheduled, obs_memory, obs_time,
                 game_id=None):
        self._api = api
        # XXX Copy to samplegame to get sample payoff reading
        self._game = paygame.samplegame_copy(rsgame.emptygame_copy(game))

        # {hprof: (lock, obs, _Record)}
        self._profiles = {}
        self._prof_lock = threading.Lock()
        self._num_running_profiles = 0
        # {prof_id: ^above^}
        self._prof_ids = {}
        self._runprof_lock = threading.Lock()
        self._pending_profiles = queue.Queue()

        self._sleep_time = sleep_time
        self._max_running = (max_scheduled * simultaneous_obs) - 1
        self._sim_id = sim_id
        self._game_id = game_id
        self._configuration = configuration
        self._obs_memory = obs_memory
        self._obs_time = obs_time
        self._simult_obs = simultaneous_obs

        self._sched = None
        self._thread_timeout_lock = threading.Lock()
        self._exception = None

    def schedule(self, profile):
        if self._exception is not None:
            raise self._exception
        hprof = gu.hash_array(profile)
        with self._prof_lock:
            val = self._profiles.setdefault(
                hprof, (threading.Lock(), queue.Queue(), _Record()))

        _, que, _ = val
        if self._num_running_profiles > self._max_running:
            _log.debug("new profile queued %s",
                       self._game.profile_to_repr(profile))
            self._pending_profiles.put((hprof, val))
        else:
            self._schedule(hprof, val)
        return _EgtaOnlinePromise(que, self)

    def game(self):
        return self._game

    def _schedule(self, hprof, val):
        """Internal implementation to actually schedule a profile

        hprof is a hashed array profile, val is the tuple stored for each
        profile. This allows differed profiles to be scheduled the same way as
        new ones."""
        profile = hprof.array
        lock, que, rec = val
        with lock:
            rec.num_req += 1

            if rec.prof_id is None:
                # Unknown id, schedule new amount
                rec.num_sched = self._simult_obs
                assignment = self._game.profile_to_repr(profile)
                _log.debug("new profile scheduled %s", assignment)
                rec.prof_id = self._sched.add_profile(
                    assignment, rec.num_sched)['id']
                self._prof_ids[rec.prof_id] = val
                with self._runprof_lock:
                    self._num_running_profiles += rec.num_sched

            elif 0 < rec.num_rec < rec.num_req and rec.num_sched == 0:
                # Id from previous data, schedule specially
                rec.num_sched = ((((rec.num_req - 1) // self._simult_obs) + 1)
                                 * self._simult_obs)
                self._prof_ids[rec.prof_id] = val
                assignment = self._game.profile_to_repr(profile)
                _log.debug(
                    "new existing data profile scheduled %s", assignment)
                self._sched.add_profile(
                    assignment, rec.num_sched)
                with self._runprof_lock:
                    self._num_running_profiles += rec.num_sched - rec.num_rec

            elif rec.num_rec < rec.num_req and rec.num_sched < rec.num_req:
                # Existing profile
                self._prof_ids[rec.prof_id] = val
                rec.num_sched += self._simult_obs
                _log.debug("old profile scheduled %s",
                           self._game.profile_to_repr(profile))
                self._sched.remove_profile(rec.prof_id)
                self._sched.add_profile(
                    self._game.profile_to_repr(profile), rec.num_sched)
                with self._runprof_lock:
                    self._num_running_profiles += self._simult_obs

    def _drain_queues(self):
        """Drain all profile queues when an assertion is thrown"""
        # This can also be thrown on keyboard interrupt
        with self._prof_lock:
            for _, que, _ in self._profiles.values():

                keep_going = [True]

                def thread_func():
                    que.get()
                    keep_going[0] = False
                    que.task_done()

                threading.Thread(target=thread_func, daemon=True).start()
                while keep_going[0]:
                    que.put(None)
                    que.join()

    def _update_counts(self):
        """Thread the constantly pings EGTA Online for updates

        This thread constantly checks an egta online scheduler for updates to
        profiles, and when found, pulls the observation data down into the
        internal structures. It's also used to schedule any profiles that
        couldn't be scheduled before because we were at max running."""
        self._thread_timeout_lock.acquire()
        try:
            while not self._thread_timeout_lock.acquire(
                    True, self._sleep_time):
                with self._runprof_lock:
                    if self._num_running_profiles == 0:
                        _log.info("nothing left to schedule")
                        continue
                # First update requirements and mark completed
                _log.info("query scheduler %d", self._sched['id'])
                info = self._sched.get_requirements()
                assert info['active'], "scheduler was deactivated"
                reqs = info['scheduling_requirements']
                _log.debug("got reqs %s", reqs)
                for req in reqs:
                    prof_id = req['id']
                    if prof_id not in self._prof_ids:
                        continue  # race condition # pragma: no cover
                    lock, que, rec = self._prof_ids[prof_id]

                    # Check if updated counts is greater than before
                    if req['current_count'] <= rec.num_rec:
                        continue
                    _log.debug("can update profile %d", prof_id)

                    with lock:
                        # If so, get observations from profile, and update data
                        # This uses fact that recent profiles are listed first,
                        # this will not work properly if this assumption is
                        # violated
                        egta_prof = self._api.get_profile(prof_id)

                        # Sometimes egta returns invalid json
                        jobs = None
                        valid = False
                        while not valid:
                            # TODO Timeouts here are probably preferred
                            jobs = egta_prof.get_info('observations')
                            valid = all(o['symmetry_groups'] is not None for o
                                        in jobs['observations'])
                        _log.debug("obs json %s",  jobs)
                        # Parse all and slice to have accurate counts
                        new_obs = self._game.samplepay_from_json(jobs)
                        # Copy so old array can be deallocated
                        new_obs = np.copy(
                            new_obs[:new_obs.shape[0] - rec.num_rec])
                        new_obs.setflags(write=False)
                        for obs in new_obs:
                            que.put(obs)
                        finished = max(min(rec.num_sched - rec.num_rec,
                                           len(new_obs)), 0)
                        rec.num_rec += len(new_obs)
                        with self._runprof_lock:
                            self._num_running_profiles -= finished

                # Now schedule any pending profiles
                while (not self._pending_profiles.empty() and
                       self._num_running_profiles <= self._max_running):
                    hprof, val = self._pending_profiles.get()
                    self._schedule(hprof, val)

        except Exception as ex:  # pragma: no cover
            self._exception = ex
            self._drain_queues()

        finally:
            try:
                self._thread_timeout_lock.release()
            except RuntimeError:  # pragma: no cover
                pass  # Don't care

    # TODO Make this an open method, and make sure it cleans up after itself
    def __enter__(self):
        # Create game to get initial profile data
        if self._game_id is not None:
            jgame = self._api.get_game(self._game_id).get_observations()
        else:
            with self._api.create_temp_game(
                    self._sim_id, self._game.num_players,
                    self._configuration) as gamea:
                _log.debug("created temp game %s %d", gamea['name'],
                           gamea['id'])
                for role, count, strats in zip(self._game.role_names,
                                               self._game.num_role_players,
                                               self._game.strat_names):
                    gamea.add_role(role, count)
                    for strat in strats:
                        gamea.add_strategy(role, strat)
                jgame = gamea.get_observations()
        assert ({(k, str(v)) for k, v in self._configuration.items()} <=
                set(map(tuple, jgame['configuration']))), \
            "games configuration didn't match"
        sim = self._api.get_simulator(self._sim_id).get_info()
        assert ('{}-{}'.format(sim['name'], sim['version']) ==
                jgame['simulator_fullname']), \
            "game didn't use the appropriate simulator"
        assert (
            {(r, c, frozenset(s)) for r, c, s
             in zip(self._game.role_names, self._game.num_role_players,
                    self._game.strat_names)} ==
            {(s['name'], s['count'], frozenset(s['strategies'])) for s
             in jgame['roles']}), \
            "game didn't have the proper role configuration"
        profiles = jgame.get('profiles', ()) or ()

        # Parse profiles
        num_profs = len(profiles)
        num_pays = 0
        for jprof in profiles:
            prof, spays = self._game.profsamplepay_from_json(jprof)
            spays.setflags(write=False)
            hprof = gu.hash_array(prof)
            que = queue.Queue()
            num_pays += len(spays)
            for pay in spays:
                que.put(pay)
            val = (threading.Lock(), que, _Record(jprof['id'], len(spays)))
            self._profiles[hprof] = val
        _log.info("found %d existing profiles with %d payoffs", num_profs,
                  num_pays)

        # Create and start scheduler
        self._sched = self._api.create_generic_scheduler(
            self._sim_id, 'egta_' + eu.random_string(20), True,
            self._obs_memory, self._game.num_players, self._obs_time,
            self._simult_obs, 1, self._configuration)
        _log.warning(
            "created scheduler %s (%d) for running simulations: "
            "https://%s/generic_schedulers/%d", self._sched['name'],
            self._sched['id'], self._api.domain, self._sched['id'])
        for role, count in zip(self._game.role_names,
                               self._game.num_role_players):
            self._sched.add_role(role, count)

        threading.Thread(target=self._update_counts, daemon=True).start()
        return self

    def __exit__(self, *args):
        if self._thread_timeout_lock.locked():
            self._thread_timeout_lock.release()
        self._sched.deactivate()
        self._drain_queues()
        _log.info("deactivated scheduler %d", self._sched['id'])


class _Record(object):
    """All mutable data associated with a profile"""

    def __init__(self, prof_id=None, num_rec=0):
        self.prof_id = prof_id
        self.num_req = 0
        self.num_sched = 0
        self.num_rec = num_rec


class _EgtaOnlinePromise(profsched.Promise):
    def __init__(self, que, sched):
        self._val = None
        self._queue = que
        self._sched = sched
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._val is None:
                self._val = self._queue.get()
                self._queue.task_done()
        if self._sched._exception is not None:
            raise self._sched._exception
        return self._val
