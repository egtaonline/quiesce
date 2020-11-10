"""python file with a bunch of utility methods"""
import random
import string


def random_string(length, choices=string.ascii_letters + string.digits):
    """Returns a random string of length length.

    Can optinally specify the characters it's drawn from."""
    return "".join(random.choice(choices) for _ in range(length))
