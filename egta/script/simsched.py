import json
import sys

from gameanalysis import gamereader

from egta import simsched


def create_scheduler(
        game='-', conf=None, command=None, buff='4096', **_):
    assert command is not None, '"command" must be specified'
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
