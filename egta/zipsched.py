"""A scheduler that gets payoffs from a local simulation"""
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import zipfile

from gameanalysis import paygame
from gameanalysis import rsgame

from egta import profsched


_log = logging.getLogger(__name__)


class ZipScheduler(profsched.Scheduler):
    """Schedule profiles using am EGTA Online zip file

    Parameters
    ----------
    game : RsGame
        A gameanalysis game that indicates how array profiles should be turned
        into json profiles.
    config : {key: value}
        A dictionary mapping string keys to values that will be passed to the
        simulator in the standard simulation spec format.
    zipcommand : string, file-like
        A zip file that follows the same semantics that EGTA Online expects.
    sleep : int, optional
        Time in seconds to wait while trying to kill threads and processes.
    max_procs : int, optional
        The maximum number of processes to spawn for simulations.
    """

    def __init__(self, game, config, zipcommand, *, sleep=1, max_procs=4):
        self._game = paygame.game_copy(rsgame.emptygame_copy(game))
        self.conf = config
        self._base = {'configuration': None, 'assignment': None}
        self.zipcommand = zipcommand
        self.sleep = sleep
        self.max_procs = max_procs

        self._running = False
        self._sim_dir = None
        self._prof_dir = None
        self._sim_root = None
        self._thread = None

        self._num = 0
        self._lock = threading.Lock()
        self._prom_queue = queue.Queue()
        self._proc_queue = queue.Queue()
        self._backup_queue = queue.Queue()

        self._exception = None

    def schedule(self, profile):
        assert self._running, \
            "can't call schedule before entering scheduler"
        if self._exception is not None:
            raise self._exception
        promise = _ZipPromise(self, profile)
        self._prom_queue.put(promise)
        self._backup_queue.put(profile)
        self._try_run()
        return promise

    def game(self):
        return self._game

    def _try_run(self):
        with self._lock:
            if self.max_procs <= self._proc_queue.qsize():
                return
            try:
                prof = self._backup_queue.get_nowait()
            except queue.Empty:
                return
            direc = os.path.join(self._prof_dir.name, str(self._num))
            os.makedirs(direc)
            self._base['assignment'] = self._game.profile_to_assignment(prof)
            with open(os.path.join(direc, 'simulation_spec.json'), 'w') as f:
                json.dump(self._base, f)
            # FIXME Schedule several at once
            proc = subprocess.Popen(
                [os.path.join('script', 'batch'), direc, '1'],
                cwd=self._sim_root)
            self._proc_queue.put((direc, proc))
            self._num += 1

    def _dequeue(self):
        """Thread used to get output from simulator

        This thread is constantly polling the simulator process and processing
        payoff data when its found."""
        try:
            while self._running:
                proc_info = self._proc_queue.get()
                if proc_info is None:
                    break
                direc, proc = proc_info
                ret = proc.wait()
                assert not ret, "process returned nonzero error code"
                self._try_run()
                obs_file = next(
                    f for f in os.listdir(direc)
                    if 'observation' in f and f.endswith('.json'))
                with open(os.path.join(direc, obs_file)) as f:
                    pay = self._game.payoff_from_json(json.load(f))
                pay.setflags(write=False)
                shutil.rmtree(direc)
                prom = self._prom_queue.get()
                _log.debug("read payoff for profile: %s",
                           self._game.profile_to_repr(prom._prof))
                prom._set(pay)
        except Exception as ex:  # pragma: no cover
            self._exception = ex
        finally:
            # Signal processes are done
            while True:
                try:
                    prom = self._prom_queue.get_nowait()
                except queue.Empty:
                    break
                if prom is None:
                    break
                prom._set(None)
            # kill processes
            while True:
                try:
                    proc_info = self._proc_queue.get_nowait()
                except queue.Empty:
                    break
                if proc_info is None:
                    break
                _, proc = proc_info
                try:
                    proc.terminate()
                except ProcessLookupError:  # pragma: no cover
                    pass  # race condition, process died
                try:
                    self._proc.wait(self.sleep)
                except subprocess.TimeoutExpired:
                    _log.warning(
                        "couldn't terminate simulation, killing it...")
                    self._proc.kill()
                try:
                    self._proc.wait(self.sleep)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    _log.error("couldn't kill simulation")

    def __enter__(self):
        self._running = True

        self._sim_dir = tempfile.TemporaryDirectory()
        self._prof_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(self.zipcommand) as zf:
            zf.extractall(self._sim_dir.name)
        sim_files = os.listdir(self._sim_dir.name)
        assert len(sim_files) == 1, "improper zip format"
        self._sim_root = os.path.join(self._sim_dir.name, sim_files[0])
        os.chmod(os.path.join(self._sim_root, 'script', 'batch'), 0o700)

        with open(os.path.join(self._sim_root, 'defaults.json')) as f:
            self._base['configuration'] = json.load(f)
        self._base['configuration'].update(self.conf)

        # We start these as daemons so that if we fail to close them for
        # whatever reason, python still exits
        self._thread = threading.Thread(target=self._dequeue, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False

        # This tells threads to die
        self._prom_queue.put(None)
        self._proc_queue.put(None)

        # Threads should be dead at this point, but we close anyways
        if self._thread is not None and self._thread.is_alive():  # pragma: no cover # noqa
            self._thread.join(self.sleep * 2 * self.max_procs)
            if self._thread.is_alive():
                _log.warning("couldn't kill dequeue thread...")

        self._sim_dir.cleanup()
        self._prof_dir.cleanup()


class _ZipPromise(profsched.Promise):
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
