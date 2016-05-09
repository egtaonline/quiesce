import itertools
import collections


class ProfileScheduler(object):
    """Class that handles scheduling profiles"""

    def __init__(self, scheduler, max_profiles, log):
        self._scheduler = scheduler
        self._max_profiles = max_profiles
        self._log = log

        self._queue = collections.deque()
        self._complete_ids = set()

    def schedule(self, profiles, numobs):
        """Add a set of profiles to schedule"""
        promise = ScheduledSet(self, numobs)
        self._queue.append(
            (promise, numobs, itertools.chain(profiles, [_END])))
        return promise

    def update(self):
        """Schedules as many profiles as possible"""
        all_profiles = self._scheduler.get_info(True).scheduling_requirements
        count_left = self._max_profiles - sum(
            req.current_count < req.requirement for req in all_profiles or ())
        all_profiles = {prof.id for prof in all_profiles}

        # Loop over necessary that we can schedule
        while count_left > 0 and self._queue:
            promise, numobs, profiles = self._queue[0]
            profile = next(profiles)

            if profile is _END:  # Reached end of list
                self._queue.popleft()
                promise._all_scheduled = True

            else:  # Process profile
                self._log.log(1, 'Scheduling profile: %s', profile)
                prof = self._scheduler.profile(profile=profile).add(numobs)
                promise._add_profile(prof)
                if prof.id not in all_profiles:
                    count_left -= 1

        # Update running ids for active checks
        all_profiles = self._scheduler.get_info(True).scheduling_requirements
        self._complete_ids = {prof.id for prof in all_profiles or ()
                              if prof.current_count >= prof.requirement}

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
            self._complete_ids.keys() <= self._scheduler._complete_ids)

    def update_count(self, new_count):
        """Request `new_count` observations"""
        self.count = new_count
        new_profiles = []
        for profile in self._complete_ids.values():
            profile.remove()
            new_profiles.append(profile.get_game_analysis_profile())
        new_profiles.append(_END)

        self._complete_ids.clear()
        self._all_scheduled = False

        self._scheduler._queue.append((self, new_count, new_profiles))

    def _add_profile(self, prof):
        self._complete_ids[prof.id] = prof


# Sentinel
def _END():
    pass
