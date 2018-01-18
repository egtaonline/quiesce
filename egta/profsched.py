"""An abstract profile scheduler"""
import abc


class Scheduler(abc.ABC):  # pragma: no cover
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    def schedule(self, profile):
        """Schedule a profile

        Return a promise for the payoff data"""
        pass

    # FIXME Add game to each scheduler
    @abc.abstractmethod
    def game(self):
        """Get the game that this scheduler returns information about"""
        pass

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, *args):
        pass


class Promise(abc.ABC):  # pragma: no cover
    """A promise for the payoff to a profile"""

    @abc.abstractmethod
    def get(self):
        """Get the payoff data blocking until available"""
        pass
