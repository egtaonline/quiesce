import argparse
import itertools
import json
import re
import requests
import sys
import tabulate
import textwrap
from datetime import datetime
from os import path

from egtaonline import api as egtaonline


_DEF_AUTH = path.join(path.dirname(path.dirname(__file__)), 'auth_token.txt')

_PARSER = argparse.ArgumentParser(prog='egta', description="""Command line
                                  access to egta online apis""")
_PARSER.add_argument('--verbose', '-v', action='count', default=0, help="""Sets
                     the verbosity of commands. Output is send to standard
                     error.""")

_PARSER_AUTH = _PARSER.add_mutually_exclusive_group()
_PARSER_AUTH.add_argument('--auth-string', '-a', metavar='<auth-string>',
                          help="""The string authorization token to connect to
                          egta online.""")
_PARSER_AUTH.add_argument('--auth-file', '-f', metavar='<auth-file>',
                          default=_DEF_AUTH, help="""Filename that just
                          contains the string of the auth token.  (default:
                          %(default)s)""")

_SUBPARSERS = _PARSER.add_subparsers(title='Subcommands', dest='command',
                                     help="""The specific aspect of the api to
                                     interact with. See each possible command
                                     for help.""")
_SUBPARSERS.required = True

_PARSER_SIM = _SUBPARSERS.add_parser('sim', description="""Operate on EGTA
                                     simulators""")
_PARSER_SIM.add_argument('sim_id', metavar='sim-id', nargs='?', help="""The
                         identifier of the simulation to get information from.
                         By default this should be the simulator id as a
                         number. If unspecified, all simulators are
                         returned.""")
_PARSER_SIM.add_argument('--version', '-n', nargs='?', metavar='version-name',
                         default=argparse.SUPPRESS, const=None, help="""If this
is specified then the `sim-id` is treated as the name of the simulator and an
optionally supplied argument is treated as the version.  If no version is
                         specified then this will try to find a single
                         simulator with the name. An error is thrown if there
                         are 0 or 2 or more simulators with the same name.""")
_PARSER_SIM.add_argument('--json', '-j', metavar='<json-file>',
                         type=argparse.FileType('r'), help="""Modify simulator
using the specified json file. By default this will add all of the roles and
strategies in the json file. If `delete` is specified, this will remove only
                         the strategies specified in the file. `-` can be used
                         to read from stdin.""")
_PARSER_SIM.add_argument('--role', '-r', metavar='<role>', help="""Modify
                         `role` of the simulator. By default this will add
                         `role` to the simulator. If `delete` is specified this
                         will remove `role` instead. If `strategy` is specified
                         see strategy.""")
_PARSER_SIM.add_argument('--strategy', '-s', metavar='<strategy>',
                         help="""Modify `strategy` of the simulator. This
                         requires that `role` is also specified. By default
                         this adds `strategy` to `role`. If `delete` is
                         specified, then this removes the strategy instead.""")
_PARSER_SIM.add_argument('--delete', '-d', action='store_true',
                         help="""Triggers removal of roles or strategies
                         instead of addition""")
_PARSER_SIM.add_argument('--zip', '-z', action='count',
                         help="""Download simulator zip file. If specified
                         once, this will download it to the current directory
                         with its proper name. If specified multiple times, it
                         will be written to stdout.""")

_PARSER_GAME = _SUBPARSERS.add_parser('game', description="""Operate on EGTA
                                      Online games.""")
_PARSER_GAME.add_argument('game_id', nargs='?', metavar='game-id', help="""The
                          identifier of the game to get data from. By default
                          this should be the game id number. If unspecified
                          this will return a list of all of the games.""")
_PARSER_GAME.add_argument('--name', '-n', action='store_true', help="""If
                          specified then get the game via its string name not
                          its id number. This is much slower than accessing via
                          id number.""")
_PARSER_GAME.add_argument('--granularity', '-g',
                          choices=('structure', 'summary', 'observations',
                                   'full'),
                          default='structure', help="""`structure`: returns the
game information but no profile information.  `summary`: returns the game
information and profiles with aggregated payoffs.  `observations`: returns the
game information and profiles with data aggregated at the observation level.
                          `full`: returns the game information and profiles
                          with complete observation information. (default:
                          %(default)s)""")
_PARSER_GAME.add_argument('--json', '-j', metavar='<json-file>',
                          type=argparse.FileType('r'), help="""Modify game
using the specified json file.  By default this will add all of the roles and
strategies in the json file. If `delete` is specified, this will remove only
the strategies specified in the file. `-` can be used to read from stdin.""")
_PARSER_GAME.add_argument('--role', '-r', metavar='<role>', help="""Modify
                          `role` of the game. By default this will add `role`
                          to the game. If `delete` is specified this will
                          remove `role` instead. If `strategy` is specified see
                          strategy.""")
_PARSER_GAME.add_argument('--count', '-c', metavar='<count>', help="""If adding
                          a role, the count of the role must be specified.""")
_PARSER_GAME.add_argument('--strategy', '-s', metavar='<strategy>',
                          help="""Modify `strategy` of the game. This requires
                          that `role` is also specified. By default this adds
                          `strategy` to `role`. If `delete` is specified, then
                          this removes the strategy instead.""")
_PARSER_GAME.add_argument('--delete', '-d', action='store_true',
                          help="""Triggers removal of roles or strategies
                          instead of addition""")

_PARSER_SCHED = _SUBPARSERS.add_parser('sched', description="""Operate on EGTA
                                       Online schedulers.""")
_PARSER_SCHED.add_argument('sched_id', nargs='?', metavar='game-id',
                           help="""The identifier of the scheduler to get data
                           from. By default this should be the scheduler id
                           number. If unspecified this will return a list of
                           all of the generic schedulers.""")
_PARSER_SCHED.add_argument('--name', '-n', action='store_true', help="""If
                           specified then get the scheduler via its string name
                           not its id number. This is much slower than
                           accessing via id number, and only works for generic
                           schedulers.""")
_PARSER_SCHED.add_argument('--verbose', '-v', action='store_true', help="""Get
                           verbose scheduler information including profile
                           information instead of just the simple output of
                           scheduler meta data.""")
_PARSER_SCHED.add_argument('--delete', '-d', action='count', help="""If
specified with a scheduler, delete it. If no scheduler is specified then filter
by inactive or complete generic schedulers that match quiesce generated names
and haven't been updated in a week and delete them.  If specified multiple
                           times, then remove all without prompt.""")
_PARSER_SCHED.add_argument('--days-old', type=int, default=7, help="""The
                           number of days stagnant (unupdated) a scheduler has
                           to be to consider it ready for removal. (default:
                           %(default)d)""")

_PARSER_SIMS = _SUBPARSERS.add_parser('sims', description="""Get information
                                      about EGTA Online simulations. These are
                                      the actual scheduled simulations instead
                                      of the simulators that generate them.""")
_PARSER_SIMS.add_argument('--folder', '-f', metavar='<folder-id>', help="""The
identifier of the simulation to get data from, normally referred to as the
folder number. If unspecified this will return a stream of json scheduler
information that can be streamed through jq to get necessary information. If
                          streamed, each result is on a new line.""")
_PARSER_SIMS.add_argument('--count', '-c', metavar='<count>', type=int,
                          help="""The maximum number of results to return of no
                          folder is specified""")
_PARSER_SIMS.add_argument('--regex', '-r', metavar='<regex>', help="""If
                          supplied will filter simulations by ones who's
                          profile string matches the given regex.""")
_PARSER_SIMS.add_argument('--page', '-p', metavar='<start-page>', default=1,
                          type=int, help="""The page to start scanning at.
                          (default: %(default)d)""")
_PARSER_SIMS.add_argument('--ascending', '-a', action='store_true',
                          help="""Return results in ascending order instead of
                          descending.""")
_PARSER_SIMS.add_argument('--sort-column', '-s',
                          choices=('job', 'folder', 'profile', 'state'),
                          default='job', help="""Column to order results by.
                          (default: %(default)s)""")


def main():
    args = _PARSER.parse_args()
    if args.auth_string is None:
        if path.isfile(args.auth_file):
            with open(args.auth_file) as auth_file:
                args.auth_string = auth_file.read().strip()
        else:
            sys.stderr.write(textwrap.fill((
                'This script needs an authorization token to access EGTA '
                'Online. By default, this script looks for file with your '
                'authorization token at "{}". See `egta --help` for other '
                'ways to specify your authorization '
                'token.').format(_DEF_AUTH)))
            sys.stderr.write('\n')
            sys.exit(1)

    with egtaonline.EgtaOnline(
            args.auth_string, logLevel=args.verbose) as api:

        if args.command == 'sim':
            if args.sim_id is None:  # Get all simulators
                json.dump(list(api.get_simulators()), sys.stdout)

            else:  # Operate on a single simulator
                # Get simulator
                if not hasattr(args, 'version'):
                    sim = api.simulator(id=int(args.sim_id))
                elif args.version is None:
                    sim = api.simulator(name=args.sim_id)
                else:
                    sim = api.simulator(name=args.sim_id, version=args.version)

                # Operate
                if args.zip:
                    info = sim.get_info()
                    url = 'https://' + api.domain + info['source']['url']
                    resp = requests.get(url)
                    resp.raise_for_status()
                    if args.zip > 1:
                        sys.stdout.buffer.write(resp.content)
                    else:
                        with open(info.name + '.zip', 'bw') as f:
                            f.write(resp.content)

                elif args.json is not None:  # Load from json
                    role_strat = json.load(args.json)
                    if args.delete:
                        sim.remove_dict(role_strat)
                    else:
                        sim.add_dict(role_strat)
                elif args.role is not None:  # Operate on single role or strat
                    if args.strategy is not None:  # Operate on strategy
                        if args.delete:
                            sim.remove_strategy(args.role, args.strategy)
                        else:
                            sim.add_strategy(args.role, args.strategy)
                    else:  # Operate on role
                        if args.delete:
                            sim.remove_role(args.role)
                        else:
                            sim.add_role(args.role)
                else:  # Return information instead
                    json.dump(sim.get_info(), sys.stdout)

        elif args.command == 'game':
            if args.game_id is None:  # Get all games
                json.dump(list(api.get_games()), sys.stdout)

            else:  # Operate on specific game
                # Get game
                if args.name:
                    game = api.game(name=args.game_id)
                else:
                    game = api.game(id=int(args.game_id))

                # Operate
                if args.json is not None:  # Load from json
                    role_strat = json.load(args.json)
                    if args.delete:
                        game.remove_dict(role_strat)
                    else:
                        game.add_dict(role_strat)

                elif args.role is not None:  # Operate on single role or strat
                    if args.strategy is not None:  # Operate on strategy
                        if args.delete:
                            game.remove_strategy(args.role, args.strategy)
                        else:
                            game.add_strategy(args.role, args.strategy)
                    else:  # Operate on role
                        if args.delete:
                            game.remove_role(args.role)
                        elif args.count:
                            game.add_role(args.role, args.count)
                        else:
                            raise ValueError('If adding a role, '
                                             'count must be specified')

                else:  # Return information instead
                    json.dump(game.get_info(
                        granularity=args.granularity), sys.stdout)

        elif args.command == 'sched':
            if args.sched_id is None:  # Get all schedulers
                scheds = api.get_generic_schedulers()
                if args.delete:
                    reg = re.compile(
                        r'.*_generic_quiesce(_[a-zA-Z]+_\d+)*_[a-zA-Z0-9]{6}')
                    now = datetime.utcnow()
                    fscheds = (s for s in scheds
                               if reg.fullmatch(s.name)
                               and args.days_old <= (now - datetime.strptime(
                                   s.updated_at, '%Y-%m-%dT%H:%M:%S.%fZ')).days
                               and (
                                   not s.active or
                                   all(p['current_count'] >= p['requirement']
                                       for p in s.get_info(True)
                                       .scheduling_requirements)))

                    if args.delete > 1:
                        inp = input('Delete all quiesce generic schedulers '
                                    'that haven\'t been updated in at least '
                                    '{} days and are either inactive or done '
                                    'scheduling [N/y]: '.format(args.days_old)
                                    ).lower()
                        if 'y' == inp:
                            for sched in fscheds:
                                sched.delete_scheduler()
                    else:
                        for sched in fscheds:
                            print(tabulate.tabulate(sched.items()))
                            inp = input('Delete Scheduler [N/y]: ').lower()
                            if 'y' == inp:
                                sched.delete_scheduler()

                else:
                    json.dump(list(scheds), sys.stdout)

            else:  # Get a single scheduler
                # Get scheduler
                if args.name:
                    sched = api.scheduler(name=args.sched_id)
                else:
                    sched = api.scheduler(id=int(args.sched_id))

                # Resolve
                if args.delete:
                    sched.delete_scheduler()
                else:
                    json.dump(sched.get_info(verbose=args.verbose), sys.stdout)

        elif args.command == 'sims':
            if args.folder is not None:  # Get info on one simulation
                sim = api.simulation(args.folder)
                json.dump(sim, sys.stdout)

            else:  # Stream simulations
                sims = api.get_simulations(page_start=args.page,
                                           asc=args.ascending,
                                           column=args.sort_column)

                # Tweak stream
                if args.regex is not None:
                    reg = re.compile(args.regex)
                    sims = (s for s in sims
                            if next(reg.finditer(s['profile']), None)
                            is not None)

                if args.count is not None:
                    sims = itertools.islice(sims, args.count)

                try:
                    for sim in sims:
                        json.dump(sim, sys.stdout)
                        sys.stdout.write('\n')
                except BrokenPipeError:
                    pass  # Don't care if stream breaks

        else:
            raise ValueError('Invalid option "{0}" specified'.format(
                args.command))
