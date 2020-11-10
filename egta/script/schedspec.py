"""Command line create and help for scheduler specifications"""
import argparse
import json

from egta import countsched
from egta import savesched
from egta.script import eosched
from egta.script import gamesched
from egta.script import simsched
from egta.script import zipsched
from egta.script import utils


def add_parser(subparsers):
    """Create scheduler spec parser"""
    parser = subparsers.add_parser(
        "spec",
        help="""Create and help for a scheduler specification""",
        description="""Create a scheduler specification for use with other
        methods. This can also be used to get information about what a
        scheduler specification should look like. A scheduler specification is
        simply a string the describes how to generate payoffs for a game.""",
    )
    parser.add_argument(
        "--save",
        metavar="<output-file>",
        default=argparse.SUPPRESS,
        help="""A file to save all sampled profile data to as a sample
        game.""",
    )
    parser.add_argument(
        "--count",
        metavar="<count>",
        type=utils.pos_int,
        default=argparse.SUPPRESS,
        help="""The number of samples to compute for
        one payoff observation.""",
    )

    types = parser.add_subparsers(
        title="schedulers",
        dest="types",
        metavar="<scheduler-type>",
        help="""The scheduler type. Create a scheduler with data from:""",
    )
    parser.types = types
    types.required = True
    for module in [gamesched, eosched, simsched, zipsched]:
        module.add_parser(types)
    parser.run = run


async def run(args):
    """Scheduler specification entry point"""
    params = dict(vars(args))
    for key in [
        "output",
        "types",
        "method",
        "recipient",
        "tag",
        "email_verbosity",
        "verbose",
    ]:
        params.pop(key)
    args.output.write(args.types)
    args.output.write(":")
    args.output.write(",".join("{}:{}".format(k, v) for k, v in params.items()))
    args.output.write("\n")


async def parse_scheduler(string):
    """Return a scheduler for a string specification"""
    stype, args = string.split(":", 1)
    args = dict(s.split(":", 1) for s in args.split(",") if s)
    count = int(args.get("count", "1"))
    save = args.pop("save", None)

    subs = argparse.ArgumentParser().add_subparsers()
    add_parser(subs)
    choices = next(iter(subs.choices.values())).types.choices

    base = await choices[stype].create_scheduler(**args)
    if save is not None:
        base = SaveWrapper(base, save)
    if count > 1:
        base = CountWrapper(base, count)
    return base


class SaveWrapper(savesched._SaveScheduler):  # pylint: disable=protected-access
    """Make save scheduler an async context manager"""

    def __init__(self, sched, dest):
        super().__init__(sched)
        self._dest = dest

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        with open(self._dest, "w") as fil:
            json.dump(self.get_game().to_json(), fil)
        await self._sched.__aexit__(*args)


class CountWrapper(countsched._CountScheduler):  # pylint: disable=protected-access
    """Make count scheduler an async context manager"""

    async def __aenter__(self):
        await self._sched.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._sched.__aexit__(*args)
