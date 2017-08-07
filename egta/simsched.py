"""A scheduler that gets payoffs from a local simulation"""
import io
import json
import logging
import queue
import subprocess
import sys
import threading
import time
import traceback

from egta import profsched


_log = logging.getLogger(__name__)


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
        self._stdin = None
        self._stdout = None
        self._lock = threading.Lock()
        self._queue = queue.Queue()

    def schedule(self, profile):
        promise = _SimPromise(self)
        with self._lock:
            self.base['assignment'] = self.serial.to_prof_json(profile)
            json.dump(self.base, self._stdin, separators=(',', ':'))
            self._stdin.write('\n')
            self._stdin.flush()
            self._queue.put(promise)
            _log.debug("sent profile: %s", profile)
        return promise

    def _dequeue(self):
        try:
            # TODO It'd be good to have this timeout to notify of problems with
            # a simulator, but I can't really do this until there's a better
            # way to interrupt the main thread.
            # TODO Similarity it'd be good to check the process for termination
            # / and error code to better notify users. We could use
            # self._proc.wait instead of time.sleep, but we'd need to handle
            # process being terminated and we'd probably want to capture stderr
            # to report to user.
            while self._running:
                line = self._stdout.readline()
                if not line:
                    time.sleep(self.sleep)
                else:
                    payoffs = self.serial.from_payoff_json(json.loads(line))
                    payoffs.setflags(write=False)
                    _log.debug("read payoff: %s", payoffs)
                    promise = self._queue.get()
                    promise._set(payoffs)
                    self._queue.task_done()
        except Exception as ex:  # pragma: no cover
            exc_type, exc_value, exc_traceback = sys.exc_info()
            _log.critical(''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)))
            raise ex

    def __enter__(self):
        self._proc = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self._stdin = io.TextIOWrapper(self._proc.stdin)
        self._stdout = io.TextIOWrapper(self._proc.stdout)

        self._running = True
        threading.Thread(target=self._dequeue, daemon=True).start()

        return self

    def __exit__(self, *args):
        self._running = False
        if self._proc is not None:
            # Kill process nicely, and then not nicely
            self._proc.terminate()
            try:
                self._proc.wait(1)
            except subprocess.TimeoutExpired:
                _log.warning("couldn't terminate simulation, killing it...")
                self._proc.kill()


class _SimPromise(profsched.Promise):
    def __init__(self, sched):
        self._event = threading.Event()
        self._sched = sched

    def _set(self, value):
        self._value = value
        self._event.set()

    def get(self):
        self._event.wait()
        return self._value
