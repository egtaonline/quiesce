"""An abstract profile scheduler"""
import abc


# FIXME Make this extend GameLike
class Scheduler(abc.ABC):  # pragma: no cover
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    async def sample_payoffs(self, profile):
        """Schedule a profile

        Return a promise for the payoff data"""
        pass
