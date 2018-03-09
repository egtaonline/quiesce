"""Python script for performing egta"""
import asyncio
import argparse
import logging
import smtplib
import sys
import traceback

from egta.script import bootstrap
from egta.script import brute
from egta.script import innerloop
from egta.script import trace


# TODO Create a scheduler that runs jobs on flux, but without going through
# egta online, potentially using spark
# FIXME Add help for schedulers

async def amain(*argv):
    parser = argparse.ArgumentParser(
        description="""Command line egta. To run, both an equilibrium finding
        method and a profile scheduler must be specified. Each element
        processes the arguments after it, e.g. `egta -a brute -b game -c` will
        pass -a to the base parser, -b to the brute equilibrium solver, and -c
        to the game profile scheduler.""")

    # Standard arguments
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', type=argparse.FileType('w'),
        default=sys.stdout, help="""The file to write the output to. (default:
        stdout)""")
    parser.add_argument(
        '--verbose', '-v', action='count', default=0, help="""Increases the
        verbosity level for standard error.""")
    parser.add_argument(
        '-e', '--email_verbosity', action='count', default=0, help="""Increases
        the verbosity level for emails.""")
    parser.add_argument(
        '-r', '--recipient', metavar='<email-address>', action='append',
        default=[], help="""Specify an email address to receive email logs at.
        Can specify multiple email addresses.""")
    parser.add_argument(
        '--tag', metavar='<tag>', help="""Specify an optional tag that will get
        appended to logs and appear in the email subject.""")
    # FIXME Add tag

    # All of the actual methods to run
    eq_methods = parser.add_subparsers(
        title='operations', dest='method', metavar='<operation>', help="""The
        operation to run on the game. Available commands are:""")
    for module in [brute, bootstrap, innerloop, trace]:
        module.add_parser(eq_methods)

    # Parse args and process
    args = parser.parse_args(argv)
    method = eq_methods.choices[args.method]

    tag = '' if args.tag is None else args.tag + ' '

    stderr_handle = logging.StreamHandler(sys.stderr)
    stderr_handle.setLevel(50 - 10 * min(args.verbose, 4))
    stderr_handle.setFormatter(logging.Formatter(
        '%(asctime)s {}{} %(message)s'.format(
            args.method, tag)))
    log_handlers = [stderr_handle]

    # Email Logging
    if args.recipient:  # pragma: no cover
        smtp_host = 'localhost'

        # We need to do this to match the from address to the local host name
        # otherwise, email logging will not work. This seems to vary somewhat
        # by machine
        with smtplib.SMTP(smtp_host) as server:
            smtp_fromaddr = 'EGTA Online <egta_online@{host}>'.format(
                host=server.local_hostname)

        email_handler = logging.handlers.SMTPHandler(
            smtp_host, smtp_fromaddr, args.recipient,
            'EGTA Status for {}{}'.format(args.method, tag))
        email_handler.setLevel(50 - args.email_verbosity * 10)
        email_handler.setFormatter(logging.Formatter(
            '%(message)s'))
        log_handlers.append(email_handler)

    logging.basicConfig(level=0, handlers=log_handlers)

    try:
        await method.run(args)

    except KeyboardInterrupt as ex:  # pragma: no cover
        logging.critical('execution interrupted by user')
        raise ex

    except Exception as ex:  # pragma: no cover
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logging.critical(''.join(traceback.format_exception(
            exc_type, exc_value, exc_traceback)))
        raise ex


def main():  # pragma: no cover
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(amain(*sys.argv[1:]))
    finally:
        loop.close()
