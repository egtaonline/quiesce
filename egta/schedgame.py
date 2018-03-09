import asyncio

import numpy as np
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import identity as idr

from egta import asyncgame


# TODO Add logging
class ReductionSchedulerGame(asyncgame.AsyncGame):
    def __init__(self, sched, red, red_players):
        super().__init__(sched.role_names, sched.strat_names,
                         sched.num_role_players)
        self._sched = sched
        self._rgame = rsgame.emptygame_copy(
            red.reduce_game(rsgame.emptygame_copy(self), red_players))
        self._red = red
        self._profiles = {}

    async def _get_game(self, profs):
        futures = []
        for prof in profs:
            hprof = utils.hash_array(prof)
            future = self._profiles.get(hprof, None)
            if future is None:
                future = asyncio.ensure_future(
                    self._sched.sample_payoffs(prof))
                self._profiles[hprof] = future
            futures.append(future)
        lpays = await asyncio.gather(*futures)
        pays = np.stack(lpays)
        return paygame.game_replace(self, profs, pays)

    def _rprofs(self, rest):
        return restrict.translate(
            self._red.expand_profiles(
                rsgame.emptygame_copy(self).restrict(rest),
                self._rgame.restrict(rest).all_profiles()),
            rest)

    async def get_restricted_game(self, rest):
        game = await self._get_game(self._rprofs(rest))
        return _ReductionGame(
            game.restrict(rest), self._red, self._rgame.num_role_players)

    async def get_deviation_game(self, rest, role_index=None):
        dprofs = self._red.expand_deviation_profiles(
            self, rest, self._rgame.num_role_players, role_index)
        rprofs = self._rprofs(rest)
        game = await self._get_game(np.concatenate([rprofs, dprofs]))
        return _ReductionGame(game, self._red, self._rgame.num_role_players)


def schedgame(sched, red=idr, red_players=None):
    return ReductionSchedulerGame(sched, red, red_players)


# TODO This should be moved to game analysis, when the appropriate contract can
# be decided upon. There are a few issues with how we treat the existence of
# profiles with whether or not deviation payoffs exists. For example, maximum
# restrictions checks that profiles are in the game, instead of whether
# deviation data exists... Which is accurate, but not what we would want for
# these style of reduced games.
class _ReductionGame(rsgame.RsGame):
    """A game with only reduced profiles

    This is a wrapper so reduced games look full."""
    def __init__(self, game, reduction, red_players):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._game = game
        self._rgame = reduction.reduce_game(game, red_players)
        self._red = reduction

    @property
    def num_complete_profiles(self):
        return self._game.num_complete_profiles

    @property
    def num_profiles(self):
        return self._game.num_profiles

    def profiles(self):
        return self._game.profiles()

    def payoffs(self):
        return self._game.payoffs()

    def deviation_payoffs(self, mix, *, jacobian=False, **kwargs):
        return self._rgame.deviation_payoffs(
            mix, jacobian=jacobian, **kwargs)

    def get_payoffs(self, profiles):
        self._game.get_payoffs(profiles)

    def max_strat_payoffs(self):
        return self._game.max_strat_payoffs()

    def min_strat_payoffs(self):
        return self._game.min_strat_payoffs()

    # FIXME These could be made more efficient by just taking the reduced game
    # at input and normalizing it here, however, this is a preliminary
    # implementation, and so not worth the efficiency.
    def normalize(self):
        return _ReductionGame(
            self._game.normalize(), self._red, self._rgame.num_role_players)

    def restrict(self, rest):
        return _ReductionGame(
            self._game.restrict(rest), self._red, self._rgame.num_role_players)

    def __contains__(self, profile):
        return profile in self._game

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self._game, self._rgame))

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._game == other._game and
                self._rgame == other._rgame)
