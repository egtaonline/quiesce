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

    # FIXME also async enter and exit, / make open close api that enter and exit just call?

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, *args):
        pass
