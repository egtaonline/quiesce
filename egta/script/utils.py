"""Utilities for command line modules"""
import argparse
import os

from gameanalysis import reduction


def add_reductions(parser):
    """Add reduction options to a parser"""
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count,role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count,role:count,...>', help="""Specify a
        hierarchical reduction.""")


def parse_reduction(game, args):
    """Parse a reduction string"""
    if args.dpr is not None:
        red_players = game.role_from_repr(args.dpr, dtype=int)
        red = reduction.deviation_preserving
    elif args.hr is not None:
        red_players = game.role_from_repr(args.hr, dtype=int)
        red = reduction.hierarchical
    else:
        red_players = None
        red = reduction.identity
    return red, red_players


def pos_int(string):
    """Type for a positive integer"""
    val = int(string)
    if val <= 0:
        raise argparse.ArgumentTypeError(
            '{:d} is an invalid positive int value'.format(string))
    return val


def check_file(string):
    """Type for a file that exists"""
    if not string == '-' and not os.path.isfile(string):
        raise argparse.ArgumentTypeError(
            '{} is not a file'.format(string))
    return string
