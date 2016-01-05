"""Python script for quiessing a game"""
import argparse
import time
import itertools
import logging
import sys
import traceback
import smtplib
import warnings
from os import path
from logging import handlers

from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import collect
from gameanalysis import reduction
from gameanalysis import nash
from gameanalysis import regret

from egtaonline import api
from egtaonline import utils
from egtaonline import containers
from egtaonline import gamesize
from egtaonline import profsched

# Game load failure is a user warning, but we don't want to process it
warnings.simplefilter('error', UserWarning)

_DEF_AUTH = path.join(path.dirname(path.dirname(__file__)), 'auth_token.txt')

_PARSER = argparse.ArgumentParser(prog='quiesce', description="""Quiesce a
                                  generic scheduler on EGTA Online.""")
_PARSER.add_argument('game', metavar='<game-id>', type=int, help="""The id of
                     the game to pull data from / to quiesce""")
_PARSER.add_argument('-p', '--max-profiles', metavar='<max-num-profiles>',
                     type=int, default=500, help="""Maximum number of profiles
                     to ever have scheduled at a time. (default:
                     %(default)s)""")
_PARSER.add_argument('-t', '--sleep-time', metavar='<sleep-time>', type=int,
                     default=300, help="""Time to wait in seconds between
                     checking EGTA Online for job completion. (default:
                     %(default)s)""")
_PARSER.add_argument('-m', '--max-subgame-size', metavar='<max-subgame-size>',
                     type=int, default=3, help="""Maximum subgame size to
                     require exploration. (default: %(default)s)""")
_PARSER.add_argument('--num-equilibria', '-n', metavar='<num-equilibria>',
                     default=1, type=int, help="""Necessary number of
equilibria to find to consider quiesced. This is useful if you want to include
a strategy that results in a trivial equilibrium e.g. a no-op. (default:
                     %(default)d)""")
# FIXME Add json input
_PARSER.add_argument('--dpr', nargs='+', metavar='<role-or-count>', default=(),
                     help="""If specified, does a dpr reduction with role
                     strategy counts.  e.g.  --dpr role1 1 role2 2 ...""")
_PARSER.add_argument('-v', '--verbose', action='count', default=0,
                     help="""Verbosity level. Two for confirmed equilibria,
                     three for major scheduling actions, four for minor
                     scheduling actions (i.e. every profile). Logging is output
                     to standard error""")
_PARSER.add_argument('-e', '--email_verbosity', action='count', default=0,
                     help="""Verbosity level for email. Two for confirmed
                     equilibria, three for everything""")
_PARSER.add_argument('-r', '--recipient', metavar='<email-address>',
                     action='append', default=[], help="""Specify an email
                     address to receive email logs at. Can specify multiple
                     email addresses.""")

_PARSER_AUTH = _PARSER.add_mutually_exclusive_group()
_PARSER_AUTH.add_argument('--auth-string', '-a', metavar='<auth-string>',
                          help="""The string authorization token to connect to
                          egta online.""")
_PARSER_AUTH.add_argument('--auth-file', '-f', metavar='<auth-file>',
                          default=_DEF_AUTH, help="""Filename that just
                          contains the string of the auth token. (default:
                          %(default)s)""")

_SCHED_GROUP = _PARSER.add_argument_group('Scheduler parameters',
                                          description="""Parameters for the
                                          scheduler.""")
_SCHED_GROUP.add_argument('-y', '--memory', metavar='<process-memory>',
                          type=int, default=4096, help="""The process memory to
                          schedule jobs with in MB.  (default: %(default)s)""")
_SCHED_GROUP.add_argument('-o', '--observation-time',
                          metavar='<observation-time>', type=int, default=600,
                          help="""The time to allow for each observation in
                          seconds. (default: %(default)s)""")
_SCHED_GROUP.add_argument('--observation-increment', '-b',
                          metavar='<observation-increment>', type=int,
                          default=1, help="""The number of observations to run
                          per simulation. (default: %(default)s)""")
_SCHED_GROUP.add_argument('--nodes', metavar='<nodes>', type=int, default=1,
                          help="""Number of nodes to run the simulation on.
                          (default: %(default)s)""")


class Quieser(object):
    """Class to manage quiesing of a scheduler"""

    def __init__(self, game_id, auth_token, max_profiles=10000, sleep_time=300,
                 subgame_limit=None, num_subgames=1, required_num_equilibria=1,
                 dpr=None, scheduler_options=collect.frozendict(), verbosity=0,
                 email_verbosity=0, recipients=[]):

        # Get api and access to standard objects
        self._log = _create_logger(
            self.__class__.__name__, verbosity, email_verbosity, recipients,
            game_id)
        self._api = api.EgtaOnline(auth_token)
        self._game_api = self._api.game(id=game_id)

        # General information about game
        game_info = self._game_api.get_info('summary')
        game_info.update(self._game_api.get_info())

        # Set up reduction
        role_counts = {r['name']: r['count'] for r in game_info['roles']}
        if dpr:
            self.reduction = reduction.DeviationPreserving(role_counts, dpr)
            role_counts = dpr
        else:
            self.reduction = reduction.Identity()

        # Set game information
        self.game = rsgame.EmptyGame(
            role_counts,
            {r['name']: r['strategies'] for r in game_info['roles']})

        # Set up scheduler
        api_scheduler = _get_scheduler(
            self._api, self._log, game_info, **scheduler_options)

        self._scheduler = profsched.ProfileScheduler(api_scheduler,
                                                     max_profiles, self._log)
        self.observation_increment = scheduler_options['observation_increment']
        self.sleep_time = sleep_time
        self.required_num_equilibria = required_num_equilibria
        self.subgame_limit = subgame_limit
        # TODO allow other functions
        self.subgame_size = gamesize.sum_strategies

    def quiesce(self):
        """Starts the process of quiescing"""
        # Dont't hardcode this
        confirmed_equilibria = containers.MixtureSet(1e-3)
        explored_subgames = containers.setset()
        backup = containers.priorityqueue()
        subgames = []  # Subgames that are running
        equilibria = []  # Equilibria that are running

        def enough_equilibria():
            return len(confirmed_equilibria) >= self.required_num_equilibria

        def add_subgame(sub):
            """Adds a subgame to the scheduler"""
            if explored_subgames.add(sub.support_set()):  # Unexplored
                self._log.debug(
                    'Exploring subgame:\n%s\n', utils.format_json(sub))
                promise = self._scheduler.schedule(
                    itertools.chain.from_iterable(
                        self.reduction.expand_profile(p)
                        for p in sub.all_profiles()),
                    self.observation_increment)
                subgames.append((sub, promise))

            else:  # Subgame already explored
                self._log.debug(
                    'Subgame already explored:\n%s\n',
                    utils.format_json(sub))

        def add_equilibrium(mixture):
            """Adds the given mixture to the scheduler"""
            self._log.debug(
                'Exploring equilibrium deviations:\n%s\n',
                utils.format_json(mixture))

            reduced_profiles = subgame.EmptySubgame(
                self.game,
                mixture.support()
            ).deviation_profiles()

            promise = self._scheduler.schedule(
                itertools.chain.from_iterable(
                    self.reduction.expand_profile(p)
                    for p, r, s in reduced_profiles),
                self.observation_increment)

            equilibria.append((mixture, promise))

        def analyze_subgame(game_data, sub, promise):
            """Computes subgame equilibrium and queues them to be scheduled"""
            if not promise.finished():  # Unfinished, so don't filter
                return True

            equilibria = list(n.trim_support() for n in nash.mixed_nash(
                subgame.subgame(game_data, sub.strategies)))
            self._log.debug(
                'Found candidate equilibria:\n%s\nin subgame:\n%s\n',
                utils.format_json(equilibria), utils.format_json(sub))
            if not equilibria:
                self._log.info(
                    'Found no equilibria in subgame:\n%s\n',
                    utils.format_json(sub))
                promise.update_count(promise.count +
                                     self.observation_increment)
                return True  # No equilibria found - keep scheduling
            else:
                for mixture in equilibria:
                    add_equilibrium(mixture)
                return False

        def analyze_equilibrium(game_data, mixture, promise):
            """Analyzes responses to an equilibrium and book keeps"""
            if not promise.finished():  # Unfinished, so don't filter
                return True

            responses = regret.mixture_deviation_gains(game_data, mixture)
            self._log.debug(
                'Responses:\n%s\nto candidate equilibrium:\n%s\n',
                utils.format_json(responses), utils.format_json(mixture))

            if all(all(d < 1e-3 for d in s.values())
                   for s in responses.values()):
                # found equilibria
                if mixture not in confirmed_equilibria:
                    confirmed_equilibria.add(mixture)
                    self._log.info(
                        'Confirmed equilibrium:\n%s\n',
                        utils.format_json(mixture))

            else:  # Queue up next subgames
                supp = mixture.support()
                sub = subgame.EmptySubgame(self.game, supp)
                subgame_size = self.subgame_size(supp,
                                                 role_counts=self.game.players)
                small_subgame = subgame_size < self.subgame_limit

                for role, role_resps in responses.items():
                    ordered = sorted(role_resps.items(), key=lambda x: x[1])
                    if ordered[-1][1] > 0:
                        # Positive best response exists for role
                        strategy, gain = ordered.pop()
                        br_subgame = sub.add_strategy(role, strategy)
                        if small_subgame:
                            add_subgame(br_subgame)
                        else:
                            backup.append((
                                (False, False, subgame_size, -gain),
                                br_subgame))

                    # Priority for backup is (not best response, not beneficial
                    # response, subgame size, deviation loss). Thus, best
                    # responses are first, then positive responses, then small
                    # subgames, then highest gain.

                    # Add the rest to the backup
                    for strat, gain in ordered:
                        dev_subgame = sub.add_strategy(role, strategy)
                        backup.append((
                            (True, gain < 0, subgame_size, -gain),
                            dev_subgame))

            return False  # Finished, so filter

        # Initialize with pure subgames
        for sub in subgame.pure_subgames(self.game):
            add_subgame(sub)

        while subgames or equilibria:  # Scheduling left to do
            self._scheduler.update()

            # See what's finished
            try:
                game_data = self.reduction.reduce_game(
                    rsgame.Game.from_json(
                        self._game_api.get_info('summary')))
            except Exception as e:
                # Sometimes getting game data fails. Just wait and try again
                self._log.debug(
                    'Encountered error getting game data: (%s) %s\nWith '
                    'traceback:\n%s\nSleeping for %d seconds...\n',
                    e.__class__.__name__, e, traceback.format_exc(),
                    self.sleep_time)
                time.sleep(self.sleep_time)
                continue

            any_finished = False

            filtered_subgames = [sub for sub in subgames
                                 if analyze_subgame(game_data, *sub)]
            any_finished |= len(filtered_subgames) != len(subgames)
            subgames = filtered_subgames

            filtered_equilibria = [mix for mix in equilibria
                                   if analyze_equilibrium(game_data, *mix)]
            any_finished |= len(filtered_equilibria) != len(equilibria)
            equilibria = filtered_equilibria

            if subgames or equilibria and not any_finished:
                # We're still waiting for jobs to complete, so take a break
                self._log.debug(
                    'Waiting %d seconds for simulations to finish...\n',
                    self.sleep_time)
                time.sleep(self.sleep_time)

            elif not subgames and not equilibria and not enough_equilibria():
                # We've finished all the required stuff, but still haven't
                # found an equilibrium, so pop a backup off
                self._log.debug('Extracting backup game\n')
                while backup and not subgames:
                    add_subgame(backup.pop()[1])

        self._log.info('Finished quiescing\nConfirmed equilibria:\n%s',
                       utils.format_json(confirmed_equilibria))


def _get_scheduler(api, log, game, process_memory=4096, observation_time=600,
                   observation_increment=1, nodes=1):
    """Creates a generic scheduler with the appropriate parameters"""

    # This is to check that the number of players in each role match, it's a
    # hacky solution, but egta doesn't expose the necessary information.
    candidate_name = '{game}_generic_quiesce_{rolecounts}'.format(
        game=game['name'],
        rolecounts='_'.join('{name}_{count:d}'.format(**r)
                            for r in game['roles']))
    sim_inst_id = game['simulator_instance_id']
    schedulers = (gs for gs in api.get_generic_schedulers() if
                  gs['simulator_instance_id'] == sim_inst_id and
                  gs['process_memory'] == process_memory and
                  gs['time_per_observation'] == observation_time and
                  gs['default_observation_requirement']
                  == observation_increment and
                  gs['observations_per_simulation']
                  == observation_increment and
                  gs['nodes'] == nodes and
                  gs['name'].startswith(candidate_name))

    try:
        sched = next(schedulers)
        # found at least one exact match so use it
        sched.update(active=1)
        log.info(
            'Using scheduler %d '
            '(http://egtaonline.eecs.umich.edu/generic_schedulers/%d)',
            sched['id'], sched['id'])
        return sched
    except StopIteration:
        pass  # Found no schedulers, that's okay, we'll create one

    # Find simulator by matching on fullname
    sim_id = utils.only(s for s in api.get_simulators()
                        if '{name}-{version}'.format(**s)
                        == game['simulator_fullname'])['id']

    # Generate a random suffix
    sched = api.simulator(id=sim_id).create_generic_scheduler(
        name='{base}_{random}'.format(
            base=candidate_name, random=utils.random_string(6)),
        active=1,
        process_memory=process_memory,
        size=game['size'],
        time_per_observation=observation_time,
        observations_per_simulation=observation_increment,
        nodes=nodes,
        default_observation_requirement=observation_increment,
        configuration=dict(game['configuration']))

    # Add roles and counts to scheduler
    for role in game['roles']:
        sched.add_role(role['name'], role['count'])

    log.info('Created scheduler %d '
             '(http://egtaonline.eecs.umich.edu/generic_schedulers/%d)',
             sched['id'], sched['id'])
    return sched


def _create_logger(name, level, email_level, recipients, game_id):
    """Returns an appropriate logger"""
    log = logging.getLogger(name)
    log.setLevel(max(40 - level * 10, 1))  # 0 is no logging
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s ({gid:d}) %(message)s'.format(gid=game_id)))
    log.addHandler(handler)

    # Email Logging
    if recipients:
        email_subject = 'EGTA Online Quiesce Status for Game {gid:d}'.format(
            gid=game_id)
        smtp_host = 'localhost'

        # We need to do this to match the from address to the local host name
        # otherwise, email logging will not work. This seems to vary somewhat
        # by machine
        # must get correct hostname to send mail
        server = smtplib.SMTP(smtp_host)
        smtp_fromaddr = 'EGTA Online <egta_online@{host}>'.format(
            host=server.local_hostname)
        server.quit()  # dummy server is now useless

        email_handler = handlers.SMTPHandler(smtp_host, smtp_fromaddr,
                                             recipients, email_subject)
        email_handler.setLevel(40 - email_level * 10)
        log.addHandler(email_handler)

    return log


def _parse_dpr(dpr_list):
    """Turn list of role counts into dictionary"""
    return {dpr_list[2 * i]: int(dpr_list[2 * i + 1])
            for i in range(len(dpr_list) // 2)}


def main():
    """Main function, declared so it doesn't have global scope"""
    args = _PARSER.parse_args()
    if args.auth_string is None:
        with open(args.auth_file) as auth_file:
            args.auth_string = auth_file.read().strip()

    quies = Quieser(
        game_id=args.game,
        auth_token=args.auth_string,
        max_profiles=args.max_profiles,
        sleep_time=args.sleep_time,
        subgame_limit=args.max_subgame_size,
        required_num_equilibria=args.num_equilibria,
        dpr=_parse_dpr(args.dpr),
        scheduler_options={
            'process_memory': args.memory,
            'observation_time': args.observation_time,
            'observation_increment': args.observation_increment,
            'nodes': args.nodes
        },
        verbosity=args.verbose,
        email_verbosity=args.email_verbosity,
        recipients=args.recipient)

    try:
        quies.quiesce()
    except Exception as e:
        quies._log.error(
            'Caught exception: (%s) %s\nWith traceback:\n%s\n',
            e.__class__.__name__, e, traceback.format_exc())


if __name__ == '__main__':
    main()
