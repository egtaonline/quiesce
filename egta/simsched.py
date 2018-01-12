"""A scheduler that gets payoffs from a local simulation"""
import json
import logging
import queue
import subprocess
import threading

from gameanalysis import paygame
from gameanalysis import rsgame

from egta import profsched


_log = logging.getLogger(__name__)


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
    sleep : int, optional
        Time in seconds to wait before checking programs stdout for results.
        Too low and a lot of cycles will be wasted querying an empty buffer,
        too fast and programs may be waiting for results while this is
        sleeping.
    """

    def __init__(self, game, config, command, sleep=1):
        self.game = paygame.game_copy(rsgame.emptygame_copy(game))
        self.base = {'configuration': config}
        self.command = command
        self.sleep = sleep

        self._running = False
        self._proc = None
        self._inthread = None
        self._outthread = None
        self._lock = threading.Lock()
        self._inqueue = queue.Queue()
        self._outqueue = queue.Queue()

        self._exception = None

    def schedule(self, profile):
        assert self._running and self._proc is not None, \
            "can't call schedule before entering scheduler"
        if self._exception is not None:
            raise self._exception
        promise = _SimPromise(self, profile)
        self._inqueue.put(promise)
        return promise

    def _enqueue(self):
        """Thread used to push lines to stdin"""
        try:
            while self._running:
                prom = self._inqueue.get()
                if prom is None:
                    return  # told to terminate
                jprof = self.game.profile_to_json(prom._prof)
                self.base['assignment'] = jprof
                json.dump(self.base, self._proc.stdin, separators=(',', ':'))
                self._proc.stdin.write('\n')
                self._proc.stdin.flush()
                self._outqueue.put(prom)
                _log.debug("sent profile: %s",
                           self.game.profile_to_repr(prom._prof))
        except Exception as ex:  # pragma: no cover
            self._exception = ex
        finally:
            while not self._inqueue.empty():
                prom = self._inqueue.get()
                prom is None or prom._set(None)

    def _dequeue(self):
        """Thread used to get output from simulator

        This thread is constantly polling the simulator process and processing
        payoff data when its found."""
        try:
            while self._running:
                # FIXME If proc is killed, it doesn't close the streams, and as
                # a result, this call blocks indefinitely. I can't figure out a
                # way around this, but as it currently stands, this thread
                # doesn't die in that case.
                line = self._proc.stdout.readline()
                if not line and self._outqueue.empty():
                    # Process closed stdout and had nothing else to run
                    return
                elif not line or self._proc.poll() is not None:
                    # Process terminated unexpectedly
                    raise RuntimeError(
                        "Process died unexpectedly with code {}".format(
                            self._proc.poll()))
                else:
                    try:
                        jpays = json.loads(line)
                    except json.JSONDecodeError:
                        raise RuntimeError(
                            "Couldn't decode \"{}\" as json".format(line))
                    payoffs = self.game.payoff_from_json(jpays)
                    payoffs.setflags(write=False)
                    promise = self._outqueue.get()
                    if promise is None:
                        return  # told to exit
                    _log.debug("read payoff for profile: %s",
                               self.game.profile_to_repr(promise._prof))
                    promise._set(payoffs)
        except Exception as ex:  # pragma: no cover
            self._exception = ex
        finally:
            while not self._outqueue.empty():
                prom = self._outqueue.get()
                prom is None or prom._set(None)

    def __enter__(self):
        self._running = True

        self._proc = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
            universal_newlines=True)

        # We start these as daemons so that if we fail to close them for
        # whatever reason, python still exits
        self._inthread = threading.Thread(target=self._enqueue, daemon=True)
        self._inthread.start()
        self._outthread = threading.Thread(target=self._dequeue, daemon=True)
        self._outthread.start()
        return self

    def __exit__(self, *args):
        self._running = False

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
                self._proc.wait(self.sleep)
            except subprocess.TimeoutExpired:
                _log.warning("couldn't terminate simulation, killing it...")
                self._proc.kill()
            try:
                self._proc.wait(self.sleep)
            except subprocess.TimeoutExpired:  # pragma: no cover
                _log.error("couldn't kill simulation")

        # Threads should be dead at this point, but we close anyways
        if self._inthread is not None and self._inthread.is_alive():  # pragma: no cover # noqa
            self._inthread.join(self.sleep * 2)
            if self._inthread.is_alive():
                _log.warning("couldn't kill dequeue thread...")
        if self._outthread is not None and self._outthread.is_alive():  # pragma: no cover # noqa
            self._outthread.join(self.sleep * 2)
            if self._outthread.is_alive():
                _log.warning("couldn't kill dequeue thread...")


class _SimPromise(profsched.Promise):
    def __init__(self, sched, prof):
        self._event = threading.Event()
        self._sched = sched
        self._prof = prof

    def _set(self, value):
        self._value = value
        self._event.set()

    def get(self):
        self._event.wait()
        if self._sched._exception is not None:
            raise self._sched._exception
        assert self._sched._running, \
            "can't get promise when scheduler is not running"
        return self._value
