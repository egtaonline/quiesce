import math
import random
import threading

from gameanalysis import utils

from egta import profsched


# TODO Add common random seed for deterministic runs.


class RsGameScheduler(profsched.Scheduler):
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
    noise_dist : (\*params) -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> \*params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    size_ratio : int, optional
        This won't generate unique parameters for more than a logarithmic
        number of profiles. This this the constant to be multiplied by the log
        of the game size.
    """

    def __init__(self, game, noise_dist=lambda: 0, param_dist=lambda: (),
                 size_ratio=200):
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._max_size = max(int(math.log(game.num_all_profiles) * size_ratio),
                             game.num_profiles)
        self._game = game
        self._params = {}
        self._lock = threading.Lock()

    def schedule(self, profile):
        index = hash(utils.hash_array(profile)) % self._max_size
        with self._lock:
            params = self._params.get(index, None)
            if params is None:
                params = self._param_dist()
                self._params[index] = params
        payoff = self._game.get_payoffs(profile) + self._noise_dist(*params)
        payoff[profile == 0] = 0
        payoff.setflags(write=False)
        return _GamePromise(payoff)

    def game(self):
        return self._game

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
    noise_dist : \*params -> ndarray, optional
        A distribution which takes the parameters for the profile and generates
        random additive payoff noise for each strategy. Strategies that aren't
        played will be zeroed out. This allows using different distributions
        for different roles.
    param_dist () -> \*params, optional
        A function for generating the parameters for each profile that govern
        how payoff noise is distributed. By default there are no parameters,
        e.g. all noise comes from the same distribution.
    """

    def __init__(self, sgame, noise_dist=lambda: 0, param_dist=lambda: ()):
        assert hasattr(sgame, 'get_sample_payoffs'), "sgame not a sample game"
        self._noise_dist = noise_dist
        self._param_dist = param_dist
        self._sgame = sgame
        self._paymap = {}
        self._lock = threading.Lock()

    def schedule(self, profile):
        hprof = utils.hash_array(profile)
        with self._lock:
            pays, params = self._paymap.get(hprof, (None, None))
            if pays is None:
                params = self._param_dist()
                pays = self._sgame.get_sample_payoffs(profile)
                self._paymap[hprof] = (pays, params)

        pay = pays[random.randrange(pays.shape[0])]
        payoff = pay + self._noise_dist(*params)
        payoff[profile == 0] = 0
        return _GamePromise(payoff)

    def game(self):
        return self._sgame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _GamePromise(profsched.Promise):
    def __init__(self, payoff):
        self._payoff = payoff

    def get(self):
        return self._payoff
