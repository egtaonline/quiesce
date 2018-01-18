"""python file with a bunch of utility methods"""
import random
import string

import numpy as np


def random_string(length, choices=string.ascii_letters + string.digits):
    """Returns a random string of length length.

    Can optinally specify the characters it's drawn from."""
    return ''.join(random.choice(choices) for _ in range(length))


def parse_reduction(game, red):
    """Parse a reduction from string format

    Format is role1:count1,role2:count2,...
    """
    reduced_players = np.empty(game.num_roles, int)
    for role_red in red.strip().split(','):
        role, count = role_red.strip().split(':')
        reduced_players[game.role_index(role.strip())] = int(count)
    return reduced_players
