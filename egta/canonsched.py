import numpy as np
from gameanalysis import rsgame

from egta import profsched


class CanonScheduler(profsched.Scheduler):
    """A scheduler that removes single strategy roles

    Parameters
    ----------
    sched : Scheduler
        The base scheduler that generates payoffs.
    """

    def __init__(self, sched):
        self._sched = sched
        role_mask = sched.game().num_role_strats > 1
        self._game = rsgame.emptygame_names(
            [r for r, m in zip(sched.game().role_names, role_mask) if m],
            sched.game().num_role_players[role_mask],
            [s for s, m in zip(sched.game().strat_names, role_mask) if m])
        self._players = sched.game().num_role_players[~role_mask]
        self._inds = np.cumsum(
            role_mask * sched.game().num_role_strats)[~role_mask]
        self._mask = role_mask.repeat(sched.game().num_role_strats)

    async def sample_payoffs(self, profile):
        full_prof = np.insert(profile, self._inds, self._players)
        full_pay = await self._sched.sample_payoffs(full_prof)
        return full_pay[self._mask]

    def game(self):
        return self._game

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._sched.__aexit__(*args)
