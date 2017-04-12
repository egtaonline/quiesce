import abc


class Scheduler(metaclass=abc.ABCMeta):  # pragma: no cover
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    def schedule(self, profile):
        """Schedule a profile

        Return a promise for the payoff data"""
        pass

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, *args):
        pass


class Promise(metaclass=abc.ABCMeta):  # pragma: no cover
    """A promise for the payoff to a profile"""

    @abc.abstractmethod
    def get(self):
        """Get the payoff data blocking until available"""
        pass
