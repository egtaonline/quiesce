"""A scheduler that gets payoffs from a local simulation"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
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
    zipf : string, file-like
        A zip file that follows the same semantics that EGTA Online expects.
    max_procs : int, optional
        The maximum number of processes to spawn for simulations.
    """

    # FIXME Add count like in eosched and use it schedule them all at once
    def __init__(self, game, conf, zipf, *, max_procs=4):
        self._game = paygame.game_copy(rsgame.emptygame_copy(game))
        self.conf = conf
        self.zipf = zipf

        self._base = {}
        self._is_open = False
        self._sim_dir = None
        self._prof_dir = None
        self._sim_root = None

        self._num = 0
        self._procs = asyncio.Semaphore(max_procs)

    # FIXME Is it possible to do more of these async
    async def sample_payoffs(self, profile):
        assert self._is_open, "must enter scheduler"
        direc = os.path.join(self._prof_dir.name, str(self._num))
        self._num += 1
        os.makedirs(direc)
        self._base['assignment'] = self._game.profile_to_assignment(profile)
        with open(os.path.join(direc, 'simulation_spec.json'), 'w') as f:
            json.dump(self._base, f)
        async with self._procs:
            proc = await asyncio.create_subprocess_shell(
                '{} {} {:d}'.format(os.path.join('script', 'batch'), direc, 1),
                cwd=self._sim_root, stderr=asyncio.subprocess.PIPE)
            _, err = await proc.communicate()
        assert proc.returncode == 0, \
            "process failed with returncode {:d} and stderr {}".format(
                proc.returncode, err)
        obs_file = next((
            f for f in os.listdir(direc)
            if 'observation' in f and f.endswith('.json')), None)
        assert obs_file is not None, \
            "simulation didn't write an observation file"
        with open(os.path.join(direc, obs_file)) as f:
            pay = self._game.payoff_from_json(json.load(f))
        pay.setflags(write=False)
        shutil.rmtree(direc)
        _log.debug("read payoff for profile: %s",
                   self._game.profile_to_repr(profile))
        return pay

    def game(self):
        return self._game

    def open(self):
        assert not self._is_open
        try:
            self._num = 0
            self._sim_dir = tempfile.TemporaryDirectory()
            self._prof_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(self.zipf) as zf:
                zf.extractall(self._sim_dir.name)
            sim_files = os.listdir(self._sim_dir.name)
            assert len(sim_files) == 1, "improper zip format"
            self._sim_root = os.path.join(self._sim_dir.name, sim_files[0])
            os.chmod(os.path.join(self._sim_root, 'script', 'batch'), 0o700)

            with open(os.path.join(self._sim_root, 'defaults.json')) as f:
                self._base['configuration'] = json.load(f)
            self._base['configuration'].update(self.conf)

            self._is_open = True
        except Exception as ex:
            self.close()
            raise ex

    def close(self):
        self._is_open = False
        self._sim_dir.cleanup()
        self._prof_dir.cleanup()

    async def __aenter__(self):
        self.open()
        return self

    async def __aexit__(self, *args):
        self.close()
