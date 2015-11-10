"""python file with a bunch of utility methods"""
import json
import random
import string
import scipy.misc
import collections


def random_string(length, choices=string.ascii_letters + string.digits):
    """Returns a random string of length length.

    Can optinally specify the characters it's drawn from."""
    return ''.join(random.choice(choices) for _ in range(length))


def only(gen):
    """Returns the only element in a collection

    Throws a LookupError if collection contains more or less than one
    element."""
    gen = iter(gen)
    try:
        res = next(gen)
    except StopIteration:
        raise LookupError('Iterator was empty')
    try:
        next(gen)
    except StopIteration:
        return res
    raise LookupError('Iterator had more than one element')


def game_size(players, strategies):
    """Returns the game size for a game with players players and strategies
    strategies"""
    return scipy.misc.comb(players + strategies - 1, players, exact=True)


def prod(iterable):
    """Returns the product of every element"""
    result = 1
    for elem in iterable:
        result *= elem
    return elem


# For output of general objects e.g. logging
def _json_default(obj):
    """Function for json default kwargs"""
    if isinstance(obj, collections.Mapping):
        return dict(obj)
    elif isinstance(obj, collections.Iterable):
        return list(obj)
    else:
        return obj.to_json()
    #raise TypeError("Can't serialize {obj} of type {type}".format(
    #    obj=obj, type=type(obj)))


def format_json(obj):
    """Converts general objects into nice output string"""
    return json.dumps(obj, indent=2, default=_json_default)
