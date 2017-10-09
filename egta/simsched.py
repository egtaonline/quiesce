"""A scheduler that gets payoffs from a local simulation"""
import io
import json
import logging
import queue
import subprocess
import threading

from egta import profsched


_log = logging.getLogger(__name__)


# FIXME Change zip wrapper to be independent scheduler
class SimulationScheduler(profsched.Scheduler):
    """Schedule profiles using a command line program

    Parameters
    ----------
    serial : GameSerializer
        A gameanalysis game serializer that indicates how array profiles should
        be turned into json profiles.
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

    def __init__(self, serial, config, command, sleep=1):
        self.serial = serial
        self.base = {'configuration': config}
        self.command = command
        self.sleep = sleep

        self._running = False
        self._proc = None
        self._thread = None
        self._stdin = None
        self._stdout = None
        self._lock = threading.Lock()
        self._queue = queue.Queue()

        self._exception = None

    def schedule(self, profile):
        promise = _SimPromise(self)
        with self._lock:
            if self._exception is not None:
                raise self._exception
            assert self._proc is not None and self._proc.poll() is None, \
                "process is not running"
            jprof = self.serial.to_prof_json(profile)
            self.base['assignment'] = jprof
            json.dump(self.base, self._stdin, separators=(',', ':'))
            self._stdin.write('\n')
            self._stdin.flush()
            self._queue.put(promise)
            _log.debug("sent profile: %s", jprof)
        return promise

    def _dequeue(self):
        """Thread used to get output from simulator

        This thread is run as a daemon thread constantly polling the simulator
        process and processing payoff data when its found."""
        try:
            while self._running:
                # This is blocking
                line = self._stdout.readline()
                ret = self._proc.poll()
                if not line and self._queue.empty():
                    # Process closed stdout and had nothing else to run
                    self._running = False
                    self._proc.terminate()
                elif ret is not None:
                    # Process terminated unexpectedly
                    raise RuntimeError(
                        "Process died unexpectedly with code {:d}".format(ret))
                else:
                    jpays = json.loads(line)
                    payoffs = self.serial.from_payoff_json(jpays)
                    payoffs.setflags(write=False)
                    _log.debug("read payoff: %s", jpays)
                    promise = self._queue.get()
                    promise._set(payoffs)
        except Exception as ex:  # pragma: no cover
            self._exception = ex
            # Drain queue to notify of exception
            while not self._queue.empty():
                self._queue.get()._set(None)

    def __enter__(self):
        self._proc = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self._stdin = io.TextIOWrapper(self._proc.stdin)
        self._stdout = io.TextIOWrapper(self._proc.stdout)

        self._thread = threading.Thread(target=self._dequeue)
        self._running = True
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False

        if self._proc is not None:
            # Kill process nicely, and then not nicely
            self._proc.terminate()
            try:
                self._proc.wait(self.sleep)
            except subprocess.TimeoutExpired:
                _log.warning("couldn't terminate simulation, killing it...")
                self._proc.kill()

        # After killing the process, the output should close and the thread
        # should die
        if self._thread is not None:
            self._thread.join(self.sleep * 2)
            if self._thread.is_alive():
                _log.warning("couldn't kill dequeue thread...")

        if self._exception is not None:
            raise self._exception


class _SimPromise(profsched.Promise):
    def __init__(self, sched):
        self._event = threading.Event()
        self._sched = sched

    def _set(self, value):
        self._value = value
        self._event.set()

    def get(self):
        self._event.wait()
        if self._sched._exception is not None:
            raise self._sched._exception
        return self._value
