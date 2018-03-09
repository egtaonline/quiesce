"""A scheduler that gets payoffs from a local simulation"""
import asyncio
import collections
import contextlib
import json
import logging
import subprocess

from gameanalysis import paygame
from gameanalysis import rsgame

from egta import profsched


class SimulationScheduler(profsched.Scheduler):
    """Schedule profiles using a command line program

    Parameters
    ----------
    game : RsGame
        A gameanalysis game that indicates how array profiles should be turned
        into json profiles.
    config : {key: value}
        A dictionary mapping string keys to values that will be passed to the
        simulator in the standard simulation spec format.
    command : [str]
        A list of strings that represents a command line program to run. This
        program must accept simulation spec files as flushed lines of input to
        standard in, and write the resulting output as an observation to
        standard out. After all input lines have been read, this must flush the
        output otherwise this could hang waiting for results that are trapped
        in a buffer.
    buff_size : int, optional
        The maximum number of bytes to send to the command at a time. The
        default should be fine for most applications, but if you know your
        machine has a larger or smaller buffer size, setting this accurately
        will prevent unnecessary blocking.
    """

    def __init__(self, game, config, command, buff_size=4096):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._game = paygame.game_copy(rsgame.emptygame_copy(game))
        self._base = {'configuration': config}
        self.command = command
        self.buff_size = buff_size

        self._is_open = False
        self._proc = None
        self._reader = None
        self._read_queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._buffer_empty = asyncio.Event()
        self._buffer_bytes = 0
        self._line_bytes = collections.deque()

        self._buffer_empty.set()

    async def sample_payoffs(self, profile):
        assert self._is_open, "not open"

        self._base['assignment'] = self._game.profile_to_json(profile)
        bprof = json.dumps(self._base, separators=(',', ':')).encode('utf8')
        size = len(bprof) + 1
        assert size < self.buff_size, \
            "profile could not be written to buffer without blocking"
        async with self._write_lock:
            self._buffer_bytes += size
            self._line_bytes.appendleft(size)
            if self._buffer_bytes >= self.buff_size:
                self._buffer_empty.clear()
                await self._buffer_empty.wait()

            got_data = asyncio.Event()
            line = [None]
            self._read_queue.put_nowait((line, got_data))

            self._proc.stdin.write(bprof)
            self._proc.stdin.write(b'\n')
            try:
                await self._proc.stdin.drain()
            except ConnectionError:  # pragma: no cover race condition
                raise RuntimeError("process died unexpectedly")

        logging.debug("scheduled profile: %s",
                      self._game.profile_to_repr(profile))
        await got_data.wait()
        if self._reader.done() and self._reader.exception() is not None:
            raise self._reader.exception()
        jpays = json.loads(line[0].decode('utf8'))
        payoffs = self._game.payoff_from_json(jpays)
        payoffs.setflags(write=False)
        logging.debug("read payoff for profile: %s",
                      self.profile_to_repr(profile))
        return payoffs

    async def _read(self):
        while True:
            line, got_data = await self._read_queue.get()
            try:
                line[0] = await self._proc.stdout.readline()
                if not len(line[0]):
                    raise RuntimeError("process died unexpectedly")
                self._buffer_bytes -= self._line_bytes.pop()
                if self._buffer_bytes < self.buff_size:  # pragma: no branch
                    self._buffer_empty.set()
            finally:
                got_data.set()

    async def open(self):
        assert not self._is_open, "can't open twice"
        assert self._proc is None
        assert self._reader is None
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            self._reader = asyncio.ensure_future(self._read())
            self._is_open = True
        except Exception as ex:
            await self.close()
            raise ex
        return self

    async def close(self):
        self._is_open = False

        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(Exception):
                await self._reader
            self._reader = None

        if self._proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._proc.wait(), 0.25)
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._proc.wait(), 0.25)
            self._proc = None

        while not self._read_queue.empty():
            self._read_queue.get_nowait()
        self._buffer_empty.set()
        self._buffer_bytes = 0
        self._line_bytes.clear()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __str__(self):
        return ' '.join(self.command)


def simsched(game, config, command, buff_size=4096):
    return SimulationScheduler(game, config, command, buff_size=buff_size)
