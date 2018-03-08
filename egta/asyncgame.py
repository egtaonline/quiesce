import abc

from gameanalysis import rsgame

# TODO There may be a better way to do this, other than raise runtime errors
# for all inappropriate methods


class AsyncGame(rsgame.RsGame):
    """An asynchronous game

    Supports asynchronous methods for ensuring particular payoff data"""

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

    # Async games don't actually have data

    @property
    def num_complete_profiles(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    @property
    def num_profiles(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def payoffs(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def profiles(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def deviation_payoffs(self, mix, *, jacobian=True, **_):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def get_payoffs(self, profiles):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def max_strat_payoffs(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def min_strat_payoffs(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def normalize(self):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def restrict(self, rest):
        raise ValueError("AsyncGames don't function in normal game contexts")

    def __contains__(self, profile):
        raise ValueError("AsyncGames don't function in normal game contexts")


class CompleteAsyncGame(AsyncGame):
    """A wrapper for a complete RsGame"""
    def __init__(self, game):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._game = game

    async def get_restricted_game(self, rest):
        return self._game.restrict(rest)

    async def get_deviation_game(self, rest, role_index=None):
        return self._game

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other) and self._game == other._game


def wrap(game):
    """Wrap a CompleteGame as an AsyncGame"""
    assert game.is_complete(), "must use a complete game"
    return CompleteAsyncGame(game)
