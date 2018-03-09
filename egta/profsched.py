"""An abstract profile scheduler"""
import abc

from gameanalysis import rsgame


class Scheduler(rsgame.GameLike):
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    async def sample_payoffs(self, profile):
        """Schedule a profile

        Return a future for the payoff data"""
        pass  # pragma: no cover
