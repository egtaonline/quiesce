"""Module for creating schedulers from game data"""
import math
import random

from gameanalysis import utils

from egta import profsched


# FIXME Add delay dist for testing unscheduled payoffs
# TODO Add common random seed for deterministic runs.


class _RsGameScheduler(profsched._Scheduler):  # pylint: disable=protected-access
    """Schedule profiles by adding noise to a game

    This scheduler will generate random parameters to assign to each profile.
    To generate a sample payoff, the scheduler will add noise to the payoff for
    that profile, generated as a function of the parameter. In order to work
    for compact game representations this will only sample a number of profiles
    logarithmic in the number of profiles in the complete game.

    Parameters
    ----------
    game : Game
        The game with payoff data to use. An exception will be thrown if a
        profile doesn't have any data in the game.
    noise_dist : (\\*params) -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> \\*params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    size_ratio : int, optional
        This won't generate unique parameters for more than a logarithmic
        number of profiles. This this the constant to be multiplied by the log
        of the game size.
    """

    def __init__(  # pragma: no branch # noqa
        self, game, noise_dist=lambda: 0, param_dist=lambda: (), size_ratio=200
    ):
        super().__init__(game.role_names, game.strat_names, game.num_role_players)
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._max_size = max(
            int(math.log(game.num_all_profiles) * size_ratio), game.num_profiles
        )
        self._game = game
        self._params = {}

    async def sample_payoffs(self, profile):
        index = hash(utils.hash_array(profile)) % self._max_size
        params = self._params.get(index, None)
        if params is None:
            params = self._param_dist()
            self._params[index] = params
        payoff = self._game.get_payoffs(profile) + self._noise_dist(*params)
        payoff[profile == 0] = 0
        payoff.setflags(write=False)
        return payoff

    def __str__(self):
        return repr(self._game)


def gamesched(game, noise_dist=lambda: 0, param_dist=lambda: (), size_ratio=200):
    """Create a game scheduler"""
    return _RsGameScheduler(
        game, noise_dist=noise_dist, param_dist=param_dist, size_ratio=size_ratio
    )


class _SampleGameScheduler(profsched._Scheduler):  # pylint: disable=protected-access
    """Schedule profiles by adding noise to a sample game

    This scheduler will generate random parameters to assign to each profile.
    To generate a sample payoff, the scheduler will sample a random payoff
    associated with the profile, and then add noise generated as a function of
    the parameter.

    Parameters
    ----------
    sgame : SampleGame
        A sample game with potentially several payoffs for each profile. An
        exception will be thrown if a profile doesn't have any data in the
        game.
    noise_dist : \\*params -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> \\*params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    """

    def __init__(  # pragma: no branch # noqa
        self, sgame, noise_dist=lambda: 0, param_dist=lambda: ()
    ):
        super().__init__(sgame.role_names, sgame.strat_names, sgame.num_role_players)
        utils.check(hasattr(sgame, "get_sample_payoffs"), "sgame not a sample game")
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._sgame = sgame
        self._paymap = {}

    async def sample_payoffs(self, profile):
        hprof = utils.hash_array(profile)
        pays, params = self._paymap.get(hprof, (None, None))
        if pays is None:
            params = self._param_dist()
            pays = self._sgame.get_sample_payoffs(profile)
            self._paymap[hprof] = (pays, params)

        pay = pays[random.randrange(pays.shape[0])]
        payoff = pay + self._noise_dist(*params)
        payoff[profile == 0] = 0
        payoff.setflags(write=False)
        return payoff

    def __str__(self):
        return repr(self._sgame)


def samplegamesched(sgame, noise_dist=lambda: 0, param_dist=lambda: ()):
    """Create a samplegame scheduler"""
    return _SampleGameScheduler(sgame, noise_dist=noise_dist, param_dist=param_dist)
