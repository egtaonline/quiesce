import numpy as np
from gameanalysis import rsgame
from gameanalysis import paygame

from egta import profsched


class SaveScheduler(profsched.Scheduler):
    """A scheduler that saves all of the payoff data for output later

    Parameters
    ----------
    game : BaseGame
        The base game of the scheduler.
    sched : Scheduler
        The base scheduler to save payoffs from
    """

    def __init__(self, sched):
        self._sched = sched
        self._game = paygame.samplegame_copy(rsgame.emptygame_copy(
            sched.game()))
        self._profiles = []
        self._payoffs = []

    async def sample_payoffs(self, profile):
        payoff = await self._sched.sample_payoffs(profile)
        self._profiles.append(profile)
        self._payoffs.append(payoff)
        return payoff

    def game(self):
        if self._profiles:
            new_profs = np.concatenate([
                self._game.flat_profiles(),
                np.stack(self._profiles)])
            new_pays = np.concatenate([
                self._game.flat_payoffs(),
                np.stack(self._payoffs)])
            self._profiles.clear()
            self._payoffs.clear()
            self._game = paygame.samplegame_replace_flat(
                self._game, new_profs, new_pays)
        return self._game
