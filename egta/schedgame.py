"""Module for handling async games defined by schedulers"""
import asyncio
import logging

import numpy as np
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import identity as idr

from egta import asyncgame


class _ReductionSchedulerGame(asyncgame._AsyncGame):  # pylint: disable=protected-access
    """A scheduler game that implicitly has a reduction"""

    def __init__(self, sched, red, red_players):
        super().__init__(sched.role_names, sched.strat_names, sched.num_role_players)
        self._sched = sched
        self._rgame = rsgame.empty_copy(
            red.reduce_game(rsgame.empty_copy(self), red_players)
        )
        self._red = red
        self._profiles = {}

    def get_game(self):
        profs = []
        pays = []
        for hprof, fpay in self._profiles.items():
            if fpay.done():
                profs.append(hprof.array)
                pays.append(fpay.result())
        return self._wrap(paygame.game_replace(self, np.stack(profs), np.stack(pays)))

    async def _get_game(self, profs):
        """Get a game from the profiles to sample"""
        futures = []
        for prof in profs:
            hprof = utils.hash_array(prof)
            future = self._profiles.get(hprof, None)
            if future is None:
                future = asyncio.ensure_future(self._sched.sample_payoffs(prof))
                self._profiles[hprof] = future
            futures.append(future)
        lpays = await asyncio.gather(*futures)
        pays = np.stack(lpays)
        return paygame.game_replace(self, profs, pays)

    def _rprofs(self, rest):
        """Get the restricted profiles for a restriction"""
        return restrict.translate(
            self._red.expand_profiles(
                rsgame.empty_copy(self).restrict(rest),
                self._rgame.restrict(rest).all_profiles(),
            ),
            rest,
        )

    def _wrap(self, game):
        """Wraps a game so it has reduced deviation payoffs"""
        return _ReductionGame(
            game, self._red.reduce_game(game, self._rgame.num_role_players)
        )

    async def get_restricted_game(self, rest):
        logging.info(
            "%s: scheduling restriction %s", self, self.restriction_to_repr(rest)
        )
        game = (await self._get_game(self._rprofs(rest))).restrict(rest)
        return self._wrap(game)

    async def get_deviation_game(self, rest, role_index=None):
        logging.info(
            "%s: scheduling deviations from %s%s",
            self,
            self.restriction_to_repr(rest),
            "" if role_index is None else " by role " + self.role_names[role_index],
        )
        dprofs = self._red.expand_deviation_profiles(
            self, rest, self._rgame.num_role_players, role_index
        )
        rprofs = self._rprofs(rest)
        game = await self._get_game(np.concatenate([rprofs, dprofs]))
        return self._wrap(game)

    def __str__(self):
        return str(self._sched)


def schedgame(sched, red=idr, red_players=None):
    """Create a scheduler game"""
    return _ReductionSchedulerGame(sched, red, red_players)


# TODO This should be moved to game analysis, when the appropriate contract can
# be decided upon. There are a few issues with how we treat the existence of
# profiles with whether or not deviation payoffs exists. For example, maximum
# restrictions checks that profiles are in the game, instead of whether
# deviation data exists... Which is accurate, but not what we would want for
# these style of reduced games.
class _ReductionGame(rsgame._RsGame):  # pylint: disable=protected-access
    """A game with only reduced profiles

    This is a wrapper so reduced games look full."""

    def __init__(self, game, rgame):
        super().__init__(game.role_names, game.strat_names, game.num_role_players)
        self._game = game
        self._rgame = rgame

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

    def deviation_payoffs(self, mixture, *, jacobian=False, **kwargs):
        return self._rgame.deviation_payoffs(mixture, jacobian=jacobian, **kwargs)

    def get_payoffs(self, profiles):
        return self._game.get_payoffs(profiles)

    def max_strat_payoffs(self):
        return self._game.max_strat_payoffs()

    def min_strat_payoffs(self):
        return self._game.min_strat_payoffs()

    def _add_constant(self, constant):
        return _ReductionGame(self._game + constant, self._rgame + constant)

    def _multiply_constant(self, constant):
        return _ReductionGame(self._game * constant, self._rgame * constant)

    def _add_game(self, othr):
        utils.check(isinstance(othr, _ReductionGame), "no efficient add")
        return _ReductionGame(
            self._game + othr._game, self._rgame + othr._rgame
        )  # pylint: disable=protected-access

    def restrict(self, restriction):
        return _ReductionGame(
            self._game.restrict(restriction), self._rgame.restrict(restriction)
        )

    def __contains__(self, profile):
        return profile in self._game

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self._game, self._rgame))

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (
            super().__eq__(othr)
            and self._game == othr._game
            and self._rgame == othr._rgame
        )
