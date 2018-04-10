"""module for bootstrapping regret and surplus"""
import asyncio

import numpy as np
from gameanalysis import utils


async def deviation_payoffs(sched, mix, num, *, boots=0, chunk_size=None):
    """Bootstrap deviation payoffs

    Parameters
    ----------
    prof_sched : Scheduler
        The scheduler to sample profiles from.
    mix : ndarray
        The mixture to calculate the regret of.
    num : int
        The number of samples to gather. Must be positive.
    boots : int, optional
        The number of bootstrap samples to take. The accuracy of bootstrap is
        independent of this number, but more will reduce the variance of the
        underlying confidence bounds. The default will compute no bootstrap
        gains.
    chunk_size : int, optional
        An implementation detail specifying how frequently profiles are
        scheduled since this algorithm inherently operates in a streaming
        manner. Ideally this number should be set such that the time to
        schedule and process chunk_size roughly equals the time for one
        simulation. It also controls how much memory this uses. By default this
        is set to ten times the number of bootstraps, or 1000 if no bootstraps
        are requested.

    Notes
    -----
    This uses memory on the order of `boots + chunk_size`. It is inefficient if
    `num` is less than boots.

    Returns
    -------
    mean_gains : ndarray (num_strats,)
        The mean deviation payoffs from the mixture.
    boot_gains : ndarray (boots, num_strats)
        The deviation payoffs for each bootstrap sample.
    """
    utils.check(num > 0, "can't schedule zero samples")
    mix = np.asarray(mix, float)
    chunk_size = chunk_size or boots * 10 or 1000
    devs = np.empty(mix.size)
    mean_devs = np.zeros(mix.size)
    boot_devs = np.zeros((boots, mix.size))
    remaining = np.empty(boots, int)
    remaining.fill(num)

    # XXX This could be made less awkward, but it would help to require python
    # 3.6
    i = 0
    futures = []

    async def update():
        """update"""
        nonlocal i
        fiter = iter(futures)
        for _ in range(len(futures) // sched.num_strats):
            for j in range(sched.num_strats):
                pay = await next(fiter)
                devs[j] = pay[j]
            np.add((devs - mean_devs) / (i + 1), mean_devs, mean_devs)
            samps = np.random.binomial(remaining, 1 / (num - i))
            np.subtract(remaining, samps, remaining)
            np.add(samps[:, None] * devs / num, boot_devs, boot_devs)
            i += 1

    left = num
    while left > 0:
        new_profs = sched.random_deviation_profiles(
            min(left, chunk_size), mix).reshape((-1, mix.size))
        left -= chunk_size
        new_futures = [asyncio.ensure_future(sched.sample_payoffs(prof))
                       for prof in new_profs]
        await update()
        futures = new_futures
    await update()

    return mean_devs, boot_devs
