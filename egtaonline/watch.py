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

from gameanalysis import profile

from egtaonline import api

# Game load failure is a user warning, but we don't want to process it
warnings.simplefilter('error', UserWarning)

_DEF_AUTH = path.join(path.dirname(path.dirname(__file__)), 'auth_token.txt')

_PARSER = argparse.ArgumentParser(prog='watch', description="""Watch a
                                  scheduler for completion or failures.""")
_PARSER.add_argument('scheduler', metavar='<scheduler-id>', type=int, help="""The id of
                     the scheduler to watch for completion / errors.""")
_PARSER.add_argument('-t', '--sleep-time', metavar='<sleep-time>', type=int,
                     default=300, help="""Time to wait in seconds between
                     checking EGTA Online for scheduler completion. (default:
                     %(default)s)""")
_PARSER.add_argument('-r', '--recipient', metavar='<email-address>',
                     action='append', default=[], help="""Specify an email
                     address to receive email logs at. Can specify multiple
                     email addresses.""")
_PARSER.add_argument('-v', '--verbose', action='store_true', help="""Output
                     verbose logging to the terminal.""")

_PARSER_AUTH = _PARSER.add_mutually_exclusive_group()
_PARSER_AUTH.add_argument('--auth-string', '-a', metavar='<auth-string>',
                          help="""The string authorization token to connect to
                          egta online.""")
_PARSER_AUTH.add_argument('--auth-file', '-f', metavar='<auth-file>',
                          default=_DEF_AUTH, help="""Filename that just
                          contains the string of the auth token. (default:
                          %(default)s)""")


def main():
    """Main function, declared so it doesn't have global scope"""
    args = _PARSER.parse_args()
    if args.auth_string is None:
        with open(args.auth_file) as auth_file:
            args.auth_string = auth_file.read().strip()

    with api.EgtaOnline(args.auth_string) as egta:

        sched = egta.scheduler(id=args.scheduler).get_info(verbose=True)
        sim_id = sched['simulator_id']
        sched_size = sched['size']

        base_job_id = next(egta.get_simulations())['job']

        # Logging
        log = logging.getLogger(__name__)
        log.setLevel(20 - 10 * args.verbose)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s ({sid:d}) %(message)s'.format(sid=args.scheduler)))
        log.addHandler(handler)

        # Email Logging
        if args.recipient:
            email_subject = 'Scheduler Watching Status for Scheduler {sid:d}'\
                .format(sid=args.scheduler)
            smtp_host = 'localhost'

            # We need to do this to match the from address to the local host
            # name otherwise, email logging will not work. This seems to vary
            # somewhat by machine must get correct hostname to send mail
            with smtplib.SMTP(smtp_host) as server:
                smtp_fromaddr = 'EGTA Online <egta_online@{host}>'.format(
                    host=server.local_hostname)

            email_handler = handlers.SMTPHandler(smtp_host, smtp_fromaddr,
                                                 args.recipient, email_subject)
            email_handler.setLevel(20)
            log.addHandler(email_handler)

        log.info('Watcher initialized for scheduler %d', args.scheduler)

        finished = False
        while True:
            try:
                reqs = sched.get_info(True).get('scheduling_requirements', ())
                finished = all(prof['current_count'] >= prof['requirement']
                               for prof in reqs)
                if finished:
                    log.info('Scheduler %d completely scheduled',
                             args.scheduler)
                    finished = True
                    sys.exit(0)

                sims = itertools.takewhile(lambda s: s['job'] > base_job_id,
                                           egta.get_simulations())
                for sim in sims:
                    # Ignore all non failed simulations
                    if sim['state'] != 'failed':
                        continue

                    log.debug('Found failed simulation %s', sim)

                    # Check if simulation uses the same simulator - slow
                    full_sim = egta.simulation(sim['folder'])
                    name_parts = full_sim['simulator_fullname'].split('-')
                    name = '-'.join(name_parts[:-1])
                    version = name_parts[-1]
                    sim_simulator_id = egta.simulator(
                        name=name, version=version).get_info()['id']
                    if sim_simulator_id != sim_id:
                        log.debug("Didn't match simulator")
                        continue

                    prof_obj = profile.Profile.from_profile_string(
                        sim['profile'])

                    # Check that player numbers match
                    sim_size = sum(sum(strats.values()) for strats
                                   in prof_obj.values())
                    if sim_size != sched_size:
                        log.debug("Didn't match size")
                        continue

                    # Check if profile matches - REALLY SLOW!
                    for prof in reqs:
                        if prof['current_count'] >= prof['requirement']:
                            continue  # Complete profile

                        ga_profile = profile.Profile.from_symmetry_groups(
                            prof.get_info()['symmetry_groups'])
                        if ga_profile == prof_obj:
                            log.info('Scheduler %d probably failing to '
                                     'complete jobs: "%s"',
                                     args.scheduler, full_sim['error_message'])
                            finished = True
                            sys.exit(1)

                    log.debug("Didn't match active profile [invalid sim]")

            except Exception as ex:
                log.info('Encountered error in watch script: (%s) %s\n'
                         'With traceback:\n%s',
                         ex.__class__.__name__, ex, traceback.format_exc())
            finally:
                if not finished:
                    log.debug('Sleeping for %s seconds', args.sleep_time)
                    time.sleep(args.sleep_time)
