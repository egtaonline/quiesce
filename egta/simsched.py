"""A scheduler that gets payoffs from a local simulation"""
import asyncio
import json
import logging
import queue
import subprocess
import threading

from gameanalysis import paygame
from gameanalysis import rsgame

from egta import profsched


_log = logging.getLogger(__name__)


# XXX There exists a coroutine process object as well that could be used in
# this circumstance, but there are issues with blocking while reading and
# writing to the subprocess streams that make threads preferable for the time
# being.


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
    """

    def __init__(self, game, config, command):
        self._game = paygame.game_copy(rsgame.emptygame_copy(game))
        self._base = {'configuration': config}
        self.command = command

        self._loop = asyncio.get_event_loop()
        self._is_open = False
        self._proc = None
        self._inthread = None
        self._outthread = None
        self._inqueue = queue.Queue()
        self._outqueue = queue.Queue()

        self._exceptions = []

    async def sample_payoffs(self, profile):
        assert self._is_open, "not open"
        if self._exceptions:
            raise self._exceptions[0]
        lock = asyncio.Lock()
        await lock.acquire()
        qitem = [profile, lock, None]
        self._inqueue.put(qitem)
        await lock.acquire()
        if self._exceptions:
            raise self._exceptions[0]
        return qitem[2]

    def game(self):
        return self._game

    def _enqueue(self):
        """Thread used to push lines to stdin"""
        lock = None
        try:
            while True:
                qitem = self._inqueue.get()
                if qitem is None:
                    return  # told to terminate
                prof, lock, _ = qitem
                assert self._is_open and not self._exceptions
                jprof = self._game.profile_to_json(prof)
                self._base['assignment'] = jprof
                json.dump(self._base, self._proc.stdin, separators=(',', ':'))
                self._proc.stdin.write('\n')
                self._proc.stdin.flush()
                assert self._is_open and not self._exceptions
                self._outqueue.put(qitem)
                lock = None
                assert self._is_open and not self._exceptions
                _log.debug("sent profile: %s",
                           self._game.profile_to_repr(prof))
        except Exception as ex:  # pragma: no cover
            self._exceptions.append(ex)
        finally:
            if lock is not None:
                self._loop.call_soon_threadsafe(lock.release)
            while True:
                try:
                    qitem = self._inqueue.get_nowait()
                    if qitem is None:
                        continue
                    _, lock, _ = qitem
                    self._loop.call_soon_threadsafe(lock.release)
                except queue.Empty:
                    break  # expected

    def _dequeue(self):
        """Thread used to get output from simulator"""
        lock = None
        try:
            while True:
                line = self._proc.stdout.readline()
                if not line and self._outqueue.empty():
                    # Process closed stdout and had nothing else to run
                    return
                assert line and self._proc.poll() is None, \
                    "Process died unexpectedly with code {}".format(
                        self._proc.poll())
                assert self._is_open and not self._exceptions
                jpays = json.loads(line)
                payoffs = self._game.payoff_from_json(jpays)
                payoffs.setflags(write=False)
                qitem = self._outqueue.get()
                if qitem is None:
                    return  # told to exit
                prof, lock, _ = qitem
                assert self._is_open and not self._exceptions
                _log.debug("read payoff for profile: %s",
                           self._game.profile_to_repr(prof))
                qitem[2] = payoffs
                self._loop.call_soon_threadsafe(lock.release)
                lock = None
                assert self._is_open and not self._exceptions
        except Exception as ex:  # pragma: no cover
            self._exceptions.append(ex)
        finally:
            if lock is not None:
                self._loop.call_soon_threadsafe(lock.release)
            while True:
                try:
                    qitem = self._outqueue.get_nowait()
                    if qitem is None:
                        continue
                    _, lock, _ = qitem
                    self._loop.call_soon_threadsafe(lock.release)
                except queue.Empty:
                    break  # expected

    def open(self):
        assert not self._is_open, "can't open twice"
        assert self._inthread is None
        assert self._outthread is None
        assert self._proc is None
        try:
            self._proc = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                universal_newlines=True)
            self._inthread = threading.Thread(
                target=self._enqueue, daemon=True)
            self._inthread.start()
            self._outthread = threading.Thread(
                target=self._dequeue, daemon=True)
            self._outthread.start()
            self._is_open = True
        except Exception as ex:
            self.close()
            raise ex

    def close(self):
        self._is_open = False

        # This tells threads to die
        self._inqueue.put(None)
        self._outqueue.put(None)

        # killing the process should close the streams and kill the threads
        if self._proc is not None and self._proc.poll() is None:
            # Kill process nicely, and then not nicely
            try:
                self._proc.terminate()
            except ProcessLookupError:  # pragma: no cover
                pass  # race condition, process died
            try:
                self._proc.wait(0.25)
            except subprocess.TimeoutExpired:
                _log.warning("couldn't terminate simulation, killing it...")
                self._proc.kill()
            try:
                self._proc.wait(0.25)
                # XXX If we get here then the streams might not be closed, and
                # we'll have a thread blocking on stdout, but we're unable to
                # close it.
            except subprocess.TimeoutExpired:  # pragma: no cover
                _log.error("couldn't kill simulation")
        self._proc = None

        # Threads should be dead at this point, but we close anyways
        if self._inthread is not None and self._inthread.is_alive():  # pragma: no cover # noqa
            self._inthread.join(0.25)
            if self._inthread.is_alive():
                _log.warning("couldn't kill enqueue thread...")
        self._inthread = None

        if self._outthread is not None and self._outthread.is_alive():  # pragma: no cover # noqa
            self._outthread.join(0.25)
            if self._outthread.is_alive():
                _log.warning("couldn't kill dequeue thread...")
        self._outthread = None

    async def __aenter__(self):
        self.open()
        return self

    async def __aexit__(self, *args):
        self.close()
