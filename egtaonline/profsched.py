import collections
import itertools

import numpy as np

from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils


class QuiesceScheduler(object):

    def __init__(self, game, reduction, prof_sched):
        self._game = game
        self._red = reduction
        self._prof_scheduler = prof_sched
        self._update_num = 0
        assert np.all(self._game.num_players == self._red.full_players)
        assert np.all(self._game.num_strategies == self._red.num_strategies)

    def schedule_subgame(self, subgame_mask, counts):
        return SubgameScheduler(self, subgame_mask, counts)

    def schedule_deviations(self, subgame_mask, counts, role_index=None):
        return DeviationScheduler(self, subgame_mask, counts, role_index)

    def update(self):
        self._update_num += 1
        return self._prof_scheduler.update()


class _BaseScheduler(object):

    def __init__(self, sched, subgame_mask, counts):
        self._sched = sched
        self.subgame_mask = subgame_mask
        self.counts = counts
        all_profs = self._profiles()
        sched._prof_scheduler.enqueue(all_profs, counts)
        self._profs = list(map(utils.hash_array, all_profs))
        self._last_update = -1
        self._last_complete = False

    def is_complete(self):
        if self._last_update != self._sched._update_num:
            self._last_update = self._sched._update_num
            self._last_complete = all(
                self._sched._prof_scheduler.get_count(phash) >= self.counts
                for phash in self._profs)
        return self._last_complete

    def update_counts(self, new_counts):
        self.counts = new_counts
        self._sched._prof_scheduler.enqueue(self._profiles(), new_counts)


class SubgameScheduler(_BaseScheduler):

    def _profiles(self):
        support = self._sched._game.role_reduce(self.subgame_mask)
        return self._sched._red.expand_profiles(subgame.translate(
            rsgame.BaseGame(self._sched._red.reduced_players,
                            support).all_profiles(),
            self.subgame_mask))

    def get_subgame(self):
        profiles = np.concatenate([phash.array[None] for phash
                                   in self._profs])
        payoffs = np.concatenate([self._sched._prof_scheduler[phash][None]
                                  for phash in self._profs])
        return subgame.subgame(self._sched._red.reduce_game(rsgame.Game(
            self._sched._game, profiles, payoffs)), self.subgame_mask)


class DeviationScheduler(_BaseScheduler):

    def __init__(self, sched, subgame_mask, counts, role_index):
        self.role_index = role_index
        super().__init__(sched, subgame_mask, counts)

    def _profiles(self):
        return self._sched._red.expand_deviation_profiles(self.subgame_mask,
                                                          self.role_index)

    def get_game(self):
        hashes = self._profs + list(map(utils.hash_array,
                                        SubgameScheduler._profiles(self)))
        profiles = np.concatenate([phash.array[None] for phash in hashes])
        payoffs = np.concatenate([self._sched._prof_scheduler[phash][None]
                                  for phash in hashes])
        return self._sched._red.reduce_game(rsgame.Game(
            self._sched._game, profiles, payoffs), True)


class ProfileScheduler(object):
    """Class that handles scheduling profiles"""

    def __init__(self, game, serializer, scheduler, max_profiles, log,
                 profiles=()):
        self._game = game
        self._serial = serializer
        self._scheduler = scheduler
        self._max_profiles = max_profiles
        self._log = log

        self._queue = collections.deque()
        # Profile hashes: utils.hash_array, Simulator ids: profile.id
        self._phash_payoffs = {}
        self._sid_payoffs = {}
        self._explored_sids = set()

        for jprof in profiles:
            phash = utils.hash_array(serializer.from_prof_symgrp(
                jprof['symmetry_groups']))
            sid = jprof['id']
            payoff = serializer.from_payoff_symgrp(jprof['symmetry_groups'])
            data = (sid, [jprof['observations_count']], payoff)
            self._phash_payoffs[phash] = data
            self._sid_payoffs[sid] = data

    def enqueue(self, profiles, numobs):
        """Add a set of profiles to schedule"""
        self._queue.append(
            (numobs, itertools.chain(profiles, [_END])))

    def update(self):
        """Schedules as many profiles as possible"""
        changed = False

        count_left = self._max_profiles
        all_profiles = self._scheduler.get_info(True).scheduling_requirements
        for prof in all_profiles or ():
            if prof.id in self._sid_payoffs:
                # If prof.id is not in self._sid_payoffs, then we haven't
                # scheduled it, so we don't care
                sid, count_ptr, payoffs = self._sid_payoffs[prof.id]
                if prof.current_count < prof.requirement:
                    count_left -= 1

                elif prof.current_count > count_ptr[0]:
                    count_ptr[0] = prof.current_count
                    np.copyto(payoffs, self._serial.from_payoff_symgrp(
                        prof.get_info().symmetry_groups))
                    changed = True

        # Loop over necessary that we can schedule
        while count_left > 0 and self._queue:
            numobs, profiles = self._queue[0]
            profile = next(profiles)

            if profile is _END:  # Reached end of list
                self._queue.popleft()

            else:  # Process profile
                phash = utils.hash_array(profile)
                if phash in self._phash_payoffs:
                    sid, [count], _ = self._phash_payoffs[phash]
                    self._explored_sids.add(sid)
                    if numobs > count:
                        sprof = self._scheduler.profile(
                            id=sid,
                            assignment=self._serial.to_prof_string(profile))
                        sprof.update_count(numobs)
                        count_left -= 1
                        changed = True
                else:
                    string = self._serial.to_prof_string(profile)
                    self._log.log(1, 'Scheduling profile: %s', string)
                    sid = self._scheduler.add_profile(string, numobs).id
                    self._explored_sids.add(sid)
                    tup = (sid, [0],
                           np.empty(self._serial.num_role_strats))
                    self._phash_payoffs[phash] = tup
                    self._sid_payoffs[sid] = tup
                    count_left -= 1
                    changed = True

        return changed

    def deactivate(self):
        """Deactivate the egta online scheduler"""
        # Must be '0' not 'False'
        self._scheduler.update(active=0)

    def __getitem__(self, phash):
        return self._phash_payoffs[phash][2]

    def __contains__(self, phash):
        return phash in self._phash_payoffs

    def get_count(self, phash):
        return (0 if phash not in self._phash_payoffs
                else self._phash_payoffs[phash][1][0])

    @property
    def num_unique_profiles(self):
        return len(self._explored_sids)

    @property
    def num_profiles(self):
        return sum(self._sid_payoffs[sid][1][0] for sid in self._explored_sids)

    def get_game(self):
        profiles = np.empty((len(self._phash_payoffs),
                             self._game.num_role_strats), int)
        payoffs = np.empty(profiles.shape, float)

        for i, (phash, (_, _, pay)) in enumerate(self._phash_payoffs.items()):
            profiles[i] = phash.array
            payoffs[i] = pay

        return rsgame.Game(self._game, profiles, payoffs)


# Sentinel
def _END():
    pass
