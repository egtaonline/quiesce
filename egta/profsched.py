"""An abstract profile scheduler"""
import abc

from gameanalysis import rsgame

# FIXME Make all classes protected

class Scheduler(rsgame._GameLike): # pylint: disable=protected-access
    """A profile scheduler

    It must be a context manager for its resources."""

    @abc.abstractmethod
    async def sample_payoffs(self, profile):
        """Schedule a profile

        Return a future for the payoff data"""
        pass  # pragma: no cover


class OpenableScheduler(Scheduler):
    """Scheduler that is opened"""
    @abc.abstractmethod
    def open(self):
        """Open the scheduler"""
        pass # pragma: no cover

    @abc.abstractmethod
    def close(self):
        """Close the scheduler"""
        pass # pragma: no cover

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class AOpenableScheduler(Scheduler):
    """Scheduler that is asynchronously opened"""
    @abc.abstractmethod
    async def aopen(self):
        """Open the scheduler"""
        pass # pragma: no cover

    @abc.abstractmethod
    async def aclose(self):
        """Close the scheduler"""
        pass # pragma: no cover

    async def __aenter__(self):
        await self.aopen()
        return self

    async def __aexit__(self, *args):
        await self.aclose()
