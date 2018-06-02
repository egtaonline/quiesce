"""A scheduler that gets payoffs from a local simulation"""
import asyncio
import itertools
import json
import logging
import os
import shutil
import tempfile
import zipfile

from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils

from egta import profsched


class _ZipScheduler(profsched._OpenableScheduler): # pylint: disable=too-many-instance-attributes,protected-access
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

    def __init__(self, game, conf, zipf, *, max_procs=4, simultaneous_obs=1):
        super().__init__(
            game.role_names, game.strat_names, game.num_role_players)
        self._game = paygame.game_copy(rsgame.empty_copy(game))
        self.conf = conf
        self.zipf = zipf

        self._extra_profs = {}
        self._base = {}
        self._count = simultaneous_obs
        self._is_open = False
        self._sim_dir = None
        self._prof_dir = None
        self._sim_root = None

        self._num = 0
        self._procs = asyncio.Semaphore(max_procs)

    async def sample_payoffs(self, profile):
        utils.check(self._is_open, 'must enter scheduler')
        hprof = utils.hash_array(profile)
        counter, queue = self._extra_profs.get(hprof, (None, None))
        if counter is not None:
            # Already scheduling some profiles
            if next(counter) >= self._count:
                self._extra_profs.pop(hprof)
            pay = await queue.get()
            logging.debug(
                'read payoff for profile: %s', self.profile_to_repr(profile))
            return pay

        else:
            # Need to schedule new profiles
            direc = os.path.join(self._prof_dir.name, str(self._num))
            self._num += 1
            queue = asyncio.Queue()
            if self._count > 1:
                self._extra_profs[hprof] = (itertools.count(2), queue)
            os.makedirs(direc)
            self._base['assignment'] = self._game.profile_to_assignment(
                profile)
            with open(os.path.join(direc, 'simulation_spec.json'),
                      'w') as fil:
                json.dump(self._base, fil)
            logging.debug(
                'scheduled %d profile%s: %s', self._count,
                '' if self._count == 1 else 's', self.profile_to_repr(profile))

            # Limit simultaneous processes
            async with self._procs:
                proc = await asyncio.create_subprocess_shell(
                    '{} {} {:d}'.format(os.path.join('script', 'batch'),
                                        direc, self._count),
                    cwd=self._sim_root, stderr=asyncio.subprocess.PIPE)
                _, err = await proc.communicate()
            utils.check(
                proc.returncode == 0,
                'process failed with returncode {:d} and stderr {}',
                proc.returncode, err)
            obs_files = (
                f for f in os.listdir(direc)
                if 'observation' in f and f.endswith('.json'))
            for _ in range(self._count):
                obs_file = next(obs_files, None)
                utils.check(
                    obs_file is not None,
                    "simulation didn't write enough observation files")
                with open(os.path.join(direc, obs_file)) as fil:
                    pay = self._game.payoff_from_json(json.load(fil))
                    pay.setflags(write=False)
                    queue.put_nowait(pay)
            obs_file = next(obs_files, None)
            utils.check(
                obs_file is None,
                'simulation wrote too many observation files')
            shutil.rmtree(direc)
            pay = queue.get_nowait()
            logging.debug('read payoff for profile: %s',
                          self.profile_to_repr(profile))
            return pay

    def open(self):
        """Open the zip scheduler"""
        utils.check(not self._is_open, "can't be open")
        try:
            self._num = 0
            self._sim_dir = tempfile.TemporaryDirectory()
            self._prof_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(self.zipf) as zfil:
                zfil.extractall(self._sim_dir.name)
            sim_files = [d for d in os.listdir(self._sim_dir.name)
                         if d not in {'__MACOSX'}]
            utils.check(
                len(sim_files) == 1,
                'improper zip format, only one file should exist in root')
            self._sim_root = os.path.join(self._sim_dir.name, sim_files[0])
            os.chmod(os.path.join(self._sim_root, 'script', 'batch'), 0o700)

            with open(os.path.join(self._sim_root, 'defaults.json')) as fil:
                self._base['configuration'] = json.load(fil).get(
                    'configuration', {})
            self._base['configuration'].update(self.conf)

            self._is_open = True
        except Exception as ex:
            self.close()
            raise ex

    def close(self):
        """Close the zip scheduler"""
        self._is_open = False
        self._sim_dir.cleanup()
        self._prof_dir.cleanup()

    def __str__(self):
        return self.zipf


def zipsched(game, conf, zipf, *, max_procs=4, simultaneous_obs=1):
    """Create a zip scheduler"""
    return _ZipScheduler(
        game, conf, zipf, max_procs=max_procs,
        simultaneous_obs=simultaneous_obs)
