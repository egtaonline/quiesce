import logging
import queue
import sys
import threading
import time
import traceback

import numpy as np
from egtaonline import api
from gameanalysis import rsgame
from gameanalysis import utils as gu

from egta import profsched
from egta import utils as eu


_log = logging.getLogger(__name__)


class EgtaScheduler(profsched.Scheduler):
    """A profile scheduler that schedules through egta online

    Parameters
    ----------
    sim_id : int
        The id of the egtaonline simulator to use.
    basegame : BasgeGame
        The gameanalysis basegame representing the game to schedule.
    serial : GameSerializer
        The gameanalysis serializer that represents how to convery array
        profiles into json profiles.
    simultanious_obs : int
        The number of simultanious observations to schedule at a time.
        Egtaonline will use this when scheduling.
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
    auth_token : str, optional
        Authorization token for egtaonline. If unspecified, egtaonline will
        search for an appropriate file containing it.
    """

    def __init__(self, sim_id, basegame, serial, simultanious_obs,
                 configuration, sleep_time, max_scheduled, obs_memory,
                 obs_time, auth_token=None):
        self._game = rsgame.basegame_copy(basegame)
        self._serial = serial

        # {hprof: (lock, obs, _Record)}
        self._profiles = {}
        self._prof_lock = threading.Lock()
        self._num_running_profiles = 0
        # {prof_id: ^above^}
        self._prof_ids = {}
        self._runprof_lock = threading.Lock()
        self._pending_profiles = queue.Queue()

        self._sleep_time = sleep_time
        self._max_running = max_scheduled - 1
        self._auth_token = auth_token
        self._sim_id = sim_id
        self._configuration = configuration
        self._obs_memory = obs_memory
        self._obs_time = obs_time
        self._simult_obs = simultanious_obs

        self._api = None
        self._sched = None
        self._running = False

    def schedule(self, profile):
        hprof = gu.hash_array(profile)
        with self._prof_lock:
            val = self._profiles.setdefault(
                hprof, (threading.Lock(), queue.Queue(), _Record()))

        _, que, _ = val
        if self._num_running_profiles > self._max_running:
            _log.debug("new profile queued %s", profile)
            self._pending_profiles.put((hprof, val))
        else:
            self._schedule(hprof, val)
        return _EgtaPromise(que, self)

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
                _log.debug("new profile scheduled %s", profile)
                rec.prof_id = self._sched.add_profile(
                    self._serial.to_prof_str(profile), rec.num_sched).id
                self._prof_ids[rec.prof_id] = val
                with self._runprof_lock:
                    self._num_running_profiles += rec.num_sched

            elif 0 < rec.num_rec < rec.num_req and rec.num_sched == 0:
                # Id from previous data, schedule specially
                rec.num_sched = ((((rec.num_req - 1) // self._simult_obs) + 1)
                                 * self._simult_obs)
                self._prof_ids[rec.prof_id] = val
                _log.debug("new existing data profile scheduled %s", profile)
                self._sched.add_profile(
                    self._serial.to_prof_str(profile), rec.num_sched)
                with self._runprof_lock:
                    self._num_running_profiles += rec.num_sched - rec.num_rec

            elif rec.num_rec < rec.num_req and rec.num_sched < rec.num_req:
                # Existing profile
                self._prof_ids[rec.prof_id] = val
                rec.num_sched += self._simult_obs
                _log.debug("old profile scheduled %s", profile)
                self._sched.remove_profile(rec.prof_id)
                self._sched.add_profile(
                    self._serial.to_prof_str(profile), rec.num_sched)
                with self._runprof_lock:
                    self._num_running_profiles += self._simult_obs

    def _update_counts(self):
        """Thread the constantly pings EGTA Online for updates

        This thread constantly checks an egta online scheduler for updates to
        profiles, and when found, pulls the observation data down into the
        internal structures. It's also used to schedule any profiles that
        couldn't be scheduled before because we were at max running."""
        try:
            while self._running:
                # First update requirements and mark completed
                _log.info("query scheduler %d", self._sched.id)
                reqs = self._sched.get_requirements().scheduling_requirements
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
                                        in jobs.observations)
                        _log.debug("obs json %s",  jobs)
                        # Parse all and slice to have accurate counts
                        new_obs = self._serial.from_samplepay_json(jobs)
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

                # Now wait some time
                time.sleep(self._sleep_time)

        except Exception as ex:  # pragma: no cover
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            raise ex

    def __enter__(self):
        self._api = api.EgtaOnlineApi(self._auth_token)
        name = 'egta_' + eu.random_string(20)

        # Create game to get initial profile data
        gamea = None
        try:
            gamea = self._api.create_game(self._sim_id, name,
                                          self._game.num_all_players,
                                          self._configuration)
            _log.debug("created temp game %s %d", name, gamea.id)
            for role, count, strats in zip(self._serial.role_names,
                                           self._game.num_players,
                                           self._serial.strat_names):
                gamea.add_role(role, count)
                for strat in strats:
                    gamea.add_strategy(role, strat)
            profiles = gamea.get_observations().profiles
        finally:
            if gamea is not None:
                gamea.destroy_game()

        # Parse profiles
        for jprof in profiles:
            prof, spays = self._serial.from_profsamplepay_json(jprof)
            spays.setflags(write=False)
            hprof = gu.hash_array(prof)
            que = queue.Queue()
            for pay in spays:
                que.put(pay)
            val = (threading.Lock(), que, _Record(jprof.id, len(spays)))
            self._profiles[hprof] = val

        # Create and start scheduler
        self._sched = self._api.create_generic_scheduler(
            self._sim_id, name, True, self._obs_memory,
            self._game.num_all_players, self._obs_time, self._simult_obs, 1,
            self._configuration)
        _log.info("created scheduler %s %d", name, self._sched.id)
        for role, count in zip(self._serial.role_names,
                               self._game.num_players):
            self._sched.add_role(role, count)
        self._running = True
        thread = threading.Thread(target=self._update_counts, daemon=True)
        thread.start()

        return self

    def __exit__(self, *args):
        self._running = False
        if self._sched is not None:
            self._sched.deactivate()
        if self._api is not None:
            self._api.close()
        _log.info("deactivated scheduler %d", self._sched.id)


class _Record(object):
    """All mutable data associated with a profile"""
    def __init__(self, prof_id=None, num_rec=0):
        self.prof_id = prof_id
        self.num_req = 0
        self.num_sched = 0
        self.num_rec = num_rec


class _EgtaPromise(profsched.Promise):
    def __init__(self, que, sched):
        self._val = None
        self._queue = que
        self._sched = sched

    def get(self):
        if self._val is None:
            self._val = self._queue.get()
        return self._val
