import logging
import threading

import numpy as np
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import identity as idr


_log = logging.getLogger(__name__)


class SchedulerGame(rsgame.CompleteGame):
    """A game with profiles backed lazily by a scheduler

    This only samples a profile once for each payoff data. To gather more
    profiles back this with a count scheduler.
    """

    def __init__(self, role_names, strat_names, role_players, sched, rest,
                 scale, offset):
        super().__init__(role_names, strat_names, role_players)
        self._sched = sched
        self._rest = rest
        self._scale = scale
        self._offset = offset

    def deviation_payoffs(self, mix, *, jacobian=False, full_jacobian=False,
                          role_index=None, **_):
        """Get deviation payoffs

        Parameters
        ----------
        mix : [float]
            The mixture to get deviations from.
        jacobian : bool, optional
            Compute the jacobian with respect to the mixture.
        full_jacobian : bool
            By default, the jacobian will only have values where the mixture
            has support. Enabling this with `jacobian` schedules the full game
            so that there is full jacobian information.
        role_index : int or None, optional
            If specified and a valid role index, this will only get deviation
            payoffs for that role.
        """
        if full_jacobian and jacobian:
            game = paygame.game_copy(self)
        else:
            game = self._sched.get_deviations(
                self._rest, mix > 0, role_index)

        if jacobian:
            dev, jac = game.deviation_payoffs(mix, jacobian=True)
            return (self._offset + self._scale * dev,
                    self._scale[:, None] * jac)
        else:
            return self._offset + self._scale * game.deviation_payoffs(mix)

    def get_payoffs(self, profs):
        profs = np.asarray(profs, int)
        pays = self._offset + self._scale * self._sched.get_payoffs(
            profs, self._rest)
        pays[profs == 0] = 0
        return pays

    def max_strat_payoffs(self):
        return self._offset + self._scale * self._sched.max_payoff[self._rest]

    def min_strat_payoffs(self):
        return self._offset + self._scale * self._sched.min_payoff[self._rest]

    def normalize(self):
        self.payoffs()  # Schedule full game
        diff = self.max_role_payoffs() - self.min_role_payoffs()
        diff[np.isclose(diff, 0)] = 1
        scale = diff.repeat(self.num_role_strats)
        offset = np.repeat(self.min_role_payoffs(), self.num_role_strats)
        return SchedulerGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._sched, self._rest, self._scale / scale,
            (self._offset - offset) / scale)

    def restrict(self, rest):
        rest = np.asarray(rest, bool)
        base = rsgame.emptygame_copy(self).restrict(rest)
        new_rest = self._rest.copy()
        new_rest[new_rest] = rest
        return SchedulerGame(
            base.role_names, base.strat_names, base.num_role_players,
            self._sched, new_rest, self._scale[rest], self._offset[rest])

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._sched == other._sched and
                np.all(self._rest == other._rest) and
                np.allclose(self._scale, other._scale) and
                np.allclose(self._offset, other._offset))

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), id(self._sched),
                     utils.hash_array(self._rest)))


def schedgame(sched, red=idr, red_players=None):
    scheduler = _Scheduler(sched, red, red_players)
    base = scheduler.red_game
    return SchedulerGame(
        base.role_names, base.strat_names, base.num_role_players, scheduler,
        np.ones(base.num_strats, bool), np.ones(base.num_strats),
        np.zeros(base.num_strats))


class _Scheduler(object):
    """The underlying scheduler object

    This maintains a mapping of profiles to payoffs such that they're easily
    queried and returned as games.
    """

    def __init__(self, sched, red=idr, red_players=None):
        self.sched = sched
        self.full_game = rsgame.emptygame_copy(sched.game())
        self.red_game = rsgame.emptygame_copy(
            red.reduce_game(self.full_game, red_players))
        self.red = red

        self.profiles = {}
        self.prof_lock = threading.Lock()

        self.min_payoff = np.full(self.red_game.num_strats, np.nan)
        self.max_payoff = np.full(self.red_game.num_strats, np.nan)
        self.min_max_lock = threading.Lock()

    def _get_game(self, eprofs):
        promises = []
        with self.prof_lock:
            for prof in eprofs:
                hprof = utils.hash_array(prof)
                prom = self.profiles.get(hprof, None)
                if prom is None:
                    prom = self.profiles[hprof] = self.sched.schedule(prof)
                promises.append(prom)
        epays = np.stack([prom.get() for prom in promises])
        game = self.red.reduce_game(
            paygame.game_replace(self.full_game, eprofs, epays),
            self.red_game.num_role_players)
        with self.min_max_lock:
            np.fmin(self.min_payoff, game.min_strat_payoffs(), self.min_payoff)
            np.fmax(self.max_payoff, game.max_strat_payoffs(), self.max_payoff)
        return game

    def get_payoffs(self, profiles, rest):
        if rest.all():
            uprofs, inv = utils.unique_axis(profiles, return_inverse=True)
            eprofs = self.red.expand_profiles(self.full_game, uprofs)
            game = self._get_game(eprofs)
            return game.payoffs()[inv]
        else:
            return self.get_payoffs(restrict.translate(
                profiles, rest), _true)[:, rest]

    def get_deviations(self, dev_rest, rest, role_index):
        rrest = dev_rest.copy()
        rrest[rrest] = rest
        rfgame = self.full_game.restrict(dev_rest)
        rrgame = self.red_game.restrict(rrest)
        dprofs = self.red.expand_deviation_profiles(
            rfgame, rest, rrgame.num_role_players, role_index)
        gprofs = self.red.expand_profiles(
            rfgame, restrict.translate(rrgame.all_profiles(), rest))
        return self._get_game(restrict.translate(
            np.concatenate([dprofs, gprofs]), dev_rest)).restrict(dev_rest)


# Used for base case
_true = np.ones(1, bool)
