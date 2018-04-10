import json
import sys

from gameanalysis import gamereader
from gameanalysis import utils

from egta import simsched


async def create_scheduler(
        game: """A file with the description of the game to generate profiles
        from. Only the basic game structure is necessary. `-` is interpreted as
        stdin."""='-',
        conf: """A file with the specific configuration to load. `-` is
        interpreted as stdin. (default: {})"""=None,
        command: """The command to run. This is space delimited, heavily
        restricting what can actually be passed. Currently the best work around
        is writing a simple shell wrapper and then executing `bash
        wrapper.sh`. (required)"""=None,
        buff: """Maximum line buffer to prevent deadlock with the subprocess.
        This default is fine unless you know what you're doing."""='65536',
        **_):
    """Get payoffs from a command line simulator. The simulator will get passed
    a new compressed simulation spec file on each line of stdin, and is
    expected to write a standard observation file to each line of stdout in the
    same order."""
    utils.check(command is not None, '"command" must be specified')
    buff_size = int(buff)

    if game == '-':
        rsgame = gamereader.load(sys.stdin)
    else:
        with open(game) as f:
            rsgame = gamereader.load(f)

    if conf is None:
        config = {}
    elif conf == '-':
        config = json.load(sys.stdin)
    else:
        with open(conf) as f:
            config = json.load(f)

    return simsched.simsched(
        rsgame, config, command.split(), buff_size=buff_size)
