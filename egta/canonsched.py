import numpy as np

from egta import profsched


class CanonScheduler(profsched.Scheduler):
    """A scheduler that removes single strategy roles

    Parameters
    ----------
    sched : Scheduler
        The base scheduler that generates payoffs.
    """

    def __init__(self, sched):
        role_mask = sched.num_role_strats > 1
        super().__init__(
            tuple(r for r, m in zip(sched.role_names, role_mask) if m),
            tuple(s for s, m in zip(sched.strat_names, role_mask) if m),
            sched.num_role_players[role_mask])
        self._sched = sched
        self._players = sched.num_role_players[~role_mask]
        self._inds = np.cumsum(
            role_mask * sched.num_role_strats)[~role_mask]
        self._mask = role_mask.repeat(sched.num_role_strats)

    async def sample_payoffs(self, profile):
        full_prof = np.insert(profile, self._inds, self._players)
        full_pay = await self._sched.sample_payoffs(full_prof)
        return full_pay[self._mask]


def canon(sched):
    return CanonScheduler(sched)
