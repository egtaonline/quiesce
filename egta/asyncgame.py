"""module for games that require asynchronous profile access"""
import abc
import asyncio

import numpy as np
from gameanalysis import rsgame
from gameanalysis import utils


class _AsyncGame(rsgame._GameLike): # pylint: disable=protected-access
    """An asynchronous game

    Supports asynchronous methods for ensuring particular payoff data"""

    @abc.abstractmethod
    def get_game(self):
        """Return all data available"""
        pass  # pragma: no cover

    @abc.abstractmethod
    async def get_restricted_game(self, rest):
        """Return a complete restricted game"""
        pass  # pragma: no cover

    @abc.abstractmethod
    async def get_deviation_game(self, rest, role_index=None):
        """Return a game with payoff data for all deviations

        If role index is specified, only deviations from that role are
        necessary."""
        pass  # pragma: no cover


class _CompleteAsyncGame(_AsyncGame):
    """A wrapper for a complete RsGame"""
    def __init__(self, game):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._game = game

    def get_game(self):
        return self._game

    async def get_restricted_game(self, rest):
        return self._game.restrict(rest)

    async def get_deviation_game(self, rest, role_index=None):
        return self._game

    def __eq__(self, other):
        return super().__eq__(other) and self._game == other._game # pylint: disable=protected-access

    def __hash__(self):
        return hash(self._game)

    def __str__(self):
        return repr(self._game)


def wrap(game):
    """Wrap a CompleteGame as an AsyncGame"""
    utils.check(game.is_complete(), 'must use a complete game')
    return _CompleteAsyncGame(game)


class _MixedAsyncGame(_AsyncGame):
    """A lazy merging of two async games"""
    def __init__(self, agame0, agame1, prob):
        super().__init__(
            agame0.role_names, agame0.strat_names, agame0.num_role_players)
        self._agame0 = agame0
        self._agame1 = agame1
        self._prob = prob

    def get_game(self):
        return rsgame.mix(
            self._agame0.get_game(), self._agame1.get_game(), self._prob)

    async def get_restricted_game(self, rest):
        game0, game1 = await asyncio.gather(
            self._agame0.get_restricted_game(rest),
            self._agame1.get_restricted_game(rest))
        return rsgame.mix(game0, game1, self._prob)

    async def get_deviation_game(self, rest, role_index=None):
        game0, game1 = await asyncio.gather(
            self._agame0.get_deviation_game(rest, role_index),
            self._agame1.get_deviation_game(rest, role_index))
        return rsgame.mix(game0, game1, self._prob)

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (
            super().__eq__(othr) and ((
                self._agame0 == othr._agame0 and
                self._agame1 == othr._agame1 and
                np.isclose(self._prob, othr._prob)
            ) or (
                self._agame0 == othr._agame1 and
                self._agame1 == othr._agame0 and
                np.isclose(self._prob, 1 - othr._prob)
            )))

    @utils.memoize
    def __hash__(self):
        return hash(frozenset([self._agame0, self._agame1]))

    def __str__(self):
        return '{} - {:g} - {}'.format(
            self._agame0, self._prob, self._agame1)


def mix(agame0, agame1, prob):
    """Mix two async games"""
    utils.check(
        rsgame.empty_copy(agame0) == rsgame.empty_copy(agame1),
        'games must have identically structure')
    return _MixedAsyncGame(agame0, agame1, prob)
