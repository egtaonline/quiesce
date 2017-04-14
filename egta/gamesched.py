import random

from gameanalysis import rsgame
from gameanalysis import utils

from egta import profsched


# TODO Add common random seed for deterministic runs.


class GameScheduler(profsched.Scheduler):
    """Schedule profiles by adding noise to a game

    This scheduler will generate random parameters to assign to each profile.
    To generate a sample payoff, the scheduler will add noise to the payoff for
    that profile, generated as a function of the parameter.

    Parameters
    ----------
    game : Game
        The game with payoff data to use. An exception will be thrown if a
        profile doesn't have any data in the game.
    noise_dist : *params -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> *params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    """

    def __init__(self, game, noise_dist=lambda: 0, param_dist=lambda: ()):
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._game = game
        self._paymap = {}

    def schedule(self, profile):
        hprof = utils.hash_array(profile)

        if hprof not in self._paymap:
            params = self._param_dist()
            pay = self._game.get_payoffs(profile)
            self._paymap[hprof] = (pay, params)
        else:
            pay, params = self._paymap[hprof]

        payoff = pay + self._noise_dist(*params)
        payoff[profile == 0] = 0
        payoff.setflags(write=False)

        return _GamePromise(payoff)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class SampleGameScheduler(profsched.Scheduler):
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
    noise_dist : *params -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> *params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    """

    def __init__(self, sgame, noise_dist=lambda: 0, param_dist=lambda: ()):
        assert isinstance(sgame, rsgame.SampleGame), "Must use a sample game"
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._sgame = sgame
        self._paymap = {}

    def schedule(self, profile):
        hprof = utils.hash_array(profile)

        if hprof not in self._paymap:
            params = self._param_dist()
            pays = self._sgame.get_sample_payoffs(profile)
            self._paymap[hprof] = (pays, params)
        else:
            pays, params = self._paymap[hprof]

        pay = pays[random.randrange(pays.shape[0])]
        payoff = pay + self._noise_dist(*params)
        payoff[profile == 0] = 0

        return _GamePromise(payoff)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class AggfnScheduler(profsched.Scheduler):
    """Schedule profiles by adding noise to an AgfnGame

    This scheduler will generate random parameters to assign to groups of
    profiles.  To generate a sample payoff, the scheduler will add noise to the
    payoff for that profile, generated as a function of the parameter for that
    profile. Noise parameters are generated to only store as much extra data as
    the compressed AgfnGame stores.

    Parameters
    ----------
    agame : AgfnGame
        The game with payoff data to use.
    noise_dist : *params -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> *params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    """

    def __init__(self, agame, noise_dist=lambda: 0, param_dist=lambda: ()):
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._agame = agame
        self._params = [None] * (agame._action_weights.size +
                                 agame._function_inputs.size +
                                 agame._function_table.size)

    def schedule(self, profile):
        index = hash(utils.hash_array(profile)) % len(self._params)
        params = self._params[index]
        if params is None:
            params = self._param_dist()
            self._params[index] = params
        payoff = self._agame.get_payoffs(profile) + self._noise_dist(*params)
        payoff[profile == 0] = 0
        return _GamePromise(payoff)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _GamePromise(profsched.Promise):
    def __init__(self, payoff):
        self._payoff = payoff

    def get(self):
        return self._payoff
