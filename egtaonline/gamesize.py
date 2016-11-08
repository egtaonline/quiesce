"""Methods for calculating game size"""
from gameanalysis import utils


def max_strategies(subgame, **_):
    """Max number of strategies per role in subgame"""
    return max(len(strats) for strats in subgame.values())


def sum_strategies(subgame, **_):
    """Sum of all strategies in each role in subgame"""
    return sum(len(strats) for strats in subgame.values())


def num_profiles(subgame, role_counts, **_):
    """Number of profiles in a subgame"""
    return utils.prod(utils.game_size(role_counts[role], len(strats))
                      for role, strats in subgame.items())
