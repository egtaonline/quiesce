import collections
import logging
import threading

import numpy as np
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import identity as idr


_log = logging.getLogger(__name__)


class SparseScheduler(object):
    """Construct games with sparse payoff data

    This abstraction supports scheduling deviations and restricted games as
    primitives, so that profile data can be reused between different restricted
    games.

    Parameters
    ----------
    prof_sched : ProfileScheduler
        The underlying profile scheduler that can be used to gather more data
        when necessary.
    game : RsGame
        The game to get profile data from.
    red : reduction, optional
        The reduction style to use.
    red_players : int or [int], optional
        The number of reduced players per role.
    """

    def __init__(self, sched, red=idr, red_players=None):
        self._sched = sched
        self._red_game = rsgame.emptygame_copy(
            red.reduce_game(sched.game(), red_players))
        self._red = red
        self._profiles = {}
        self._lock = threading.Lock()

    def game(self):
        return self._sched.game()

    def _get_game(self, profiles, count):
        """Get a reduced game with the specified profiles and count"""
        promises = []
        with self._lock:
            for prof in profiles:
                hprof = utils.hash_array(prof)
                prof = self._profiles.setdefault(
                    hprof, _PayoffData(self._sched, prof))
                promise = prof.schedule(count)
                promises.append(promise)
        payoffs = np.concatenate([prom.get()[None] for prom in promises])
        return self._red.reduce_game(paygame.game_replace(
            self.game(), profiles, payoffs), self._red_game.num_role_players)

    def _restricted_game_profiles(self, rest):
        """All profiles for the restricted game"""
        return self._red.expand_profiles(self.game(), restrict.translate(
            self._red_game.restrict(rest).all_profiles(), rest))

    def get_restricted_game(self, rest, count):
        """Get game with all payoff data

        The returned game is complete for the restriction specified."""
        _log.info('scheduling %d samples from restricted game %s',
                  count, self.game().restriction_to_repr(rest))
        return self._get_game(self._restricted_game_profiles(rest), count)

    def get_deviations(self, rest, count, role_index=None):
        """Get game with all payoff data and deviations

        The returned game has complete payoff data and deviation data for the
        restricted game specified."""
        _log.info(
            'scheduling %d samples of deviations from %s%s',
            count, self.game().restriction_to_repr(rest),
            '' if role_index is None else ' with role {}'.format(
                self.game().role_names[role_index]))
        rest_profiles = self._restricted_game_profiles(rest)
        dev_profiles = self._red.expand_deviation_profiles(
            self.game(), rest, self._red_game.num_role_players,
            role_index)
        return self._get_game(np.concatenate((rest_profiles, dev_profiles)),
                              count)

    def get_data(self):
        """Get all current payoff data"""
        profs, pays = [], []
        with self._lock:
            for paydata in self._profiles.values():
                with paydata._lock:
                    profs.append(paydata._profile)
                    pays.append(paydata._payoffs)
        return self._red.reduce_game(
            paygame.game_replace(self.game(), np.stack(profs), np.stack(pays)),
            self._red_game.num_role_players)


class _PayoffData(object):
    """Payoff data for a profile"""

    def __init__(self, sched, profile):
        self._sched = sched
        self._profile = profile
        self._payoffs = np.zeros(profile.size, float)
        self._count = 0

        self._lock = threading.Lock()
        self._promises = collections.deque()

    def schedule(self, n):
        with self._lock:
            for _ in range(n - self._count - len(self._promises)):
                prom = self._sched.schedule(self._profile)
                self._promises.append(_UpdatePromise(self, prom))
        return _PayoffPromise(self, n)


class _UpdatePromise(object):
    def __init__(self, data, promise):
        self._data = data
        self._promise = promise
        self._lock = threading.Lock()
        self._updated = False

    def wait(self):
        with self._lock:
            if not self._updated:
                pay = self._promise.get()
                with self._data._lock:
                    self._data._count += 1
                    self._data._payoffs += (
                        (pay - self._data._payoffs) / self._data._count)
                self._updated = True


class _PayoffPromise(object):
    def __init__(self, data, n):
        self._data = data
        self._n = n

    def get(self):
        while True:
            with self._data._lock:
                if self._n <= self._data._count:
                    return self._data._payoffs
                updater = self._data._promises[0]
            updater.wait()
