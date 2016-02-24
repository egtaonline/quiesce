import itertools
import collections


class ProfileScheduler(object):
    """Class that handles scheduling profiles"""

    def __init__(self, scheduler, max_profiles, log):
        self._scheduler = scheduler
        self._max_profiles = max_profiles
        self._log = log

        self._queue = collections.deque()
        self._running_ids = set()

    def schedule(self, profiles, numobs):
        """Add a set of profiles to schedule"""
        promise = ScheduledSet(self, numobs)
        self._queue.append(
            (promise, numobs, profiles))
        return promise

    def update(self):
        """Schedules as many profiles as possible"""
        count = self._scheduler.num_running_profiles()

        # Loop over necessary that we can schedule
        while count < self._max_profiles and self._queue:
            promise, numobs, profiles = self._queue[0]

            for profile in itertools.islice(itertools.chain(profiles, [None]),
                                            self._max_profiles - count):
                if profile is None:  # Reached end of list
                    self._queue.popleft()
                    promise._all_scheduled = True

                else:  # Process profile
                    self._log.log(1, 'Scheduling profile: %s', profile)
                    prof = self._scheduler.profile(profile=profile).add(numobs)
                    promise._add_profile(prof)

            # Update count
            count = self._scheduler.num_running_profiles()

        # Update running ids for active checks
        self._running_ids = \
            {p['id'] for p in self._scheduler.running_profiles()}

    def deactivate(self):
        """Deactivate the egta online scheduler"""
        # Must be '0' not 'False'
        self._scheduler.update(active=0)


class ScheduledSet(object):
    def __init__(self, profile_scheduler, count):
        self._scheduler = profile_scheduler
        self.count = count
        self._complete_ids = {}
        self._all_scheduled = False

    def finished(self):
        """Returns true if this was entirely scheduled and finished"""
        return (
            self._all_scheduled and
            self._complete_ids.keys().isdisjoint(self._scheduler._running_ids))

    def update_count(self, new_count):
        """Request `new_count` observations"""
        self.count = new_count
        new_profiles = []
        for profile in self._complete_ids.values():
            profile.remove()
            new_profiles.append(profile.get_game_analysis_profile())

        self._complete_ids.clear()
        self._all_scheduled = False

        self._scheduler._queue.append((self, new_count, new_profiles))

    def _add_profile(self, prof):
        self._complete_ids[prof['id']] = prof
