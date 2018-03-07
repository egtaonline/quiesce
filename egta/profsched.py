"""An abstract profile scheduler"""
import abc


class Scheduler(abc.ABC):  # pragma: no cover
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    async def sample_payoffs(self, profile):
        """Schedule a profile

        Return a promise for the payoff data"""
        pass

    @abc.abstractmethod
    def game(self):
        """Get the game that this scheduler returns information about"""
        pass
