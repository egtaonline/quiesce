"""Utilities for command line modules"""
import argparse
import inspect
import itertools
import json
import shutil
import textwrap

from gameanalysis import reduction

from egta import countsched
from egta import savesched
from egta.script import eosched
from egta.script import gamesched
from egta.script import simsched
from egta.script import zipsched


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


_TYPES = {
    'zip': zipsched.create_scheduler,
    'sim': simsched.create_scheduler,
    'game': gamesched.create_scheduler,
    'eo': eosched.create_scheduler,
}


async def parse_scheduler(string):
    """Return a scheduler for a string specification"""
    stype, args, *_ = itertools.chain(string.split(':', 1), [''])
    args = dict(s.split(':', 1) for s in args.split(',') if s)
    base = await _TYPES[stype](**args)
    if 'save' in args:
        base = SaveWrapper(base, args['save'])
    count = int(args.get('count', '1'))
    if count > 1:
        base = CountWrapper(base, count)
    return base


def add_scheduler_epilog(parser, help_indent=15):
    """Epilog for scheduler format"""
    width, _ = shutil.get_terminal_size((80, 20))
    indent = ' ' * help_indent
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.description = textwrap.fill(
        ' '.join(parser.description.split()), width=width)

    epilog = ['scheduler specification:', '']
    epilog.append(textwrap.fill(
        ' '.join("""
A scheduler specification tells egta how to get samples of profiles. Their
format looks like `<type>:[param:value][,param:value]...`. Each scheduler type
and their supported parameters are listed below. For example:""".split()),
        width=width))
    epilog.extend(['', 'eo:game:827,mem:2048,time:600', ''])
    epilog.append(textwrap.fill(' '.join("""
Represents an egtaonline scheduler for game 827 with 2048 MB and 600 seconds
per observation.""".split())))
    epilog.append('')
    for name, func in itertools.chain(
            [('<global>', _global)], sorted(_TYPES.items())):
        start = '  ' + name
        desc = inspect.getdoc(func) or ''
        desc = ' '.join(desc.split())
        if len(start) < help_indent - 1:
            epilog.extend(textwrap.wrap(
                '{:{}}{}'.format(start, help_indent, desc), width=width,
                subsequent_indent=indent))
        else:
            epilog.append(start)
            epilog.extend(textwrap.wrap(
                desc, width=width, initial_indent=indent,
                subsequent_indent=indent))

        for pname, param in inspect.signature(func).parameters.items():
            if pname == '_':
                continue
            start = '    ' + pname
            desc = ''
            if param.annotation is not inspect._empty:  # pragma: no branch pylint: disable=protected-access
                desc += param.annotation
            if param.default is not None:
                desc += ' (default: {})'.format(param.default)
            desc = ' '.join(desc.split())
            if len(start) < help_indent - 1:
                epilog.extend(textwrap.wrap(
                    '{:{}}{}'.format(start, help_indent, desc), width=width,
                    subsequent_indent=indent))
            else:
                epilog.append(start)
                epilog.extend(textwrap.wrap(
                    desc, width=width, initial_indent=indent,
                    subsequent_indent=indent))
        epilog.append('')
    parser.epilog = '\n'.join(epilog)


def _global(
        # pylint: disable-msg=unused-argument
        count: """The number of samples to compute for one payoff
        observations.""" = '1',
        save: """A file to save all sampled profile data to as a sample
        game.""" = None):
    """Global parameters that apply to all schedulers."""
    pass  # pragma: no cover


class SaveWrapper(savesched.SaveScheduler):
    """Make save scheduler an async context manager"""
    def __init__(self, sched, dest):
        super().__init__(sched)
        self._dest = dest

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        with open(self._dest, 'w') as fil:
            json.dump(self.get_game().to_json(), fil)
        await self._sched.__aexit__(*args)


class CountWrapper(countsched.CountScheduler):
    """Make count scheduler an async context manager"""
    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._sched.__aexit__(*args)
