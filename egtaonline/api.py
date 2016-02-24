"""Python package to handle python interface to egta online api"""
import requests
import json
import logging
import sys
import functools
import itertools

from lxml import etree

from gameanalysis import collect
from gameanalysis import profile

from egtaonline import utils

# Decorator to update object id if unknown as most api calls require an id


def _requires_id(func):
    @functools.wraps(func)
    def get_id_first(self, *args, **kwargs):
        if 'id' not in self:
            self.get_info()
        return func(self, *args, **kwargs)
    return get_id_first


def _encode_data(data):
    """Takes data in nested dictionary form, and converts it for egta

    All dictionary keys must be strings. This call is non destructive.
    """
    encoded = {}
    for k, val in data.items():
        if isinstance(val, dict):
            for inner_key, inner_val in _encode_data(val).items():
                encoded['{0}[{1}]'.format(k, inner_key)] = inner_val
        else:
            encoded[k] = val
    return encoded


class EgtaOnline(object):
    """Class to wrap egtaonline api"""

    def __init__(self, auth_token, domain='egtaonline.eecs.umich.edu',
                 logLevel=0):
        self._domain = domain
        self._session = requests.Session()
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(40 - logLevel * 10)
        self._log.addHandler(logging.StreamHandler(sys.stderr))

        # This authenticates us for the duration of the session
        self._session.get('https://{domain}'.format(domain=self._domain),
                          data={'auth_token': auth_token})

    def close(self):
        """Closes the active session"""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._session.close()

    def _request(self, verb, api, data=collect.frozendict()):
        """Convenience method for making requests"""
        data = _encode_data(data)
        url = 'https://{domain}/api/v3/{endpoint}'.format(
            domain=self._domain, endpoint=api)
        self._log.info('%s request to %s with data %s', verb, url, data)
        return self._session.request(verb, url, data=data)

    def _non_api_request(self, verb, api, data=collect.frozendict()):
        data = _encode_data(data)
        url = 'https://{domain}/{endpoint}'.format(
            domain=self._domain, endpoint=api)
        self._log.info('%s request to %s with data %s', verb, url, data)
        return self._session.request(verb, url, data=data)

    def simulator(self, *args, **kwargs):
        """Get a simulator with given properties

        Specifying `id` is the most efficient, but a `name` and optionally a
        `version` if several simulators with the same name exist will also
        work."""
        return Simulator(*args, _api=self, **kwargs)

    def get_simulators(self):
        """Get all known simulators"""
        resp = self._request('get', 'simulators')
        resp.raise_for_status()
        return map(self.simulator, json.loads(resp.text)['simulators'])

    def scheduler(self, *args, **kwargs):
        """Get a scheduler with given properties

        Specifying `id` is the most efficient, but `name` will also suffice."""
        return Scheduler(*args, _api=self, **kwargs)

    def get_generic_schedulers(self):
        """Get a generator of all known generic schedulers"""
        resp = self._request('get', 'generic_schedulers')
        resp.raise_for_status()
        return map(self.scheduler, json.loads(resp.text)['generic_schedulers'])

    def game(self, *args, **kwargs):
        """Get a game with given properties

        Specifying `id` is the most efficient, but `name` will also suffice."""
        return Game(*args, _api=self, **kwargs)

    def get_games(self):
        """Get a generator of all of the game structures"""
        resp = self._request('get', 'games')
        resp.raise_for_status()
        return map(self.game, json.loads(resp.text)['games'])

    def profile(self, *args, **kwargs):
        """Get a profile with given properties

        `scheduler_id` must be specified for most actions. `id` will make most
        calls significantly more efficient. `id`s can be found with a
        scheduler's `get_info` when verbose=True."""
        return Profile(*args, _api=self, **kwargs)

    _mapping = {
        'job': 'job_id',
        'folder': None,
        'profile': 'profiles.assignment',
        'state': 'state'
    }
    _rows = ['state', 'profile', 'folder', 'job']

    @staticmethod
    def _parse(res):
        """Converts N/A to `nan` and otherwise tries to parse integers"""
        try:
            return int(res)
        except ValueError:
            if res.lower() == 'n/a':
                return float('nan')
            else:
                return res

    def get_simulations(self, page_start=1, asc=False, column='job_id'):
        """Get information about current simulations

        `page_start` must be at least 1. `column` should be
        one of 'job', 'folder', 'profile', or 'state'."""
        column = self._mapping.get(column, column)
        data = {
            'direction': 'ASC' if asc else 'DESC'
        }
        if column is not None:
            data['sort'] = column
        for page in itertools.count(page_start):
            data['page'] = page
            resp = self._non_api_request('get', 'simulations', data=data)
            resp.raise_for_status()
            rows = etree.HTML(resp.text).xpath('//tbody/tr')
            if not rows:
                break  # Empty page implies we're done
            for row in rows:
                res = (self._parse(''.join(e.itertext()))
                       for e in row.getchildren())
                yield dict(zip(self._rows, res))

    def simulation(self, folder):
        resp = self._non_api_request(
            'get',
            'simulations/{folder}'.format(folder=folder))
        resp.raise_for_status()
        info = etree.HTML(resp.text).xpath(
            '//div[@class="show_for simulation"]/p')
        parsed = (''.join(e.itertext()).split(':', 1) for e in info)
        return {key.lower().replace(' ', '_'): self._parse(val.strip())
                for key, val in parsed}


class Simulator(dict):
    """Get information about and modify EGTA Online Simulators"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api = self.pop('_api')

    def get_info(self):
        """Return information about this simulator

        If the id is unknown this will search all simulators for one with the
        same name and optionally version. If version is unspecified, but only
        one simulator with that name exists, this lookup should still succeed.
        This returns a new simulator object, but will update the id of the
        current simulator if it was undefined."""
        if 'id' in self:
            resp = self._api._request(
                'get', 'simulators/{sim:d}.json'.format(sim=self['id']))
            resp.raise_for_status()
            result = self._api.simulator(json.loads(resp.text))
            self['id'] = result['id']

        elif 'version' in self:
            result = utils.only(
                sim for sim in self._api.get_simulators()
                if sim['name'] == self['name']
                and sim['version'] == self['version'])

        else:
            result = utils.only(
                sim for sim in self._api.get_simulators()
                if sim['name'] == self['name'])

        return result

    @_requires_id
    def add_role(self, role):
        """Adds a role to the simulator"""
        resp = self._api._request(
            'post',
            'simulators/{sim:d}/add_role.json'.format(sim=self['id']),
            data={'role': role})
        resp.raise_for_status()

    @_requires_id
    def remove_role(self, role):
        """Removes a role from the simulator"""
        resp = self._api._request(
            'post',
            'simulators/{sim:d}/remove_role.json'.format(sim=self['id']),
            data={'role': role})
        resp.raise_for_status()

    @_requires_id
    def add_strategy(self, role, strategy):
        """Adds a strategy to the simulator"""
        resp = self._api._request(
            'post',
            'simulators/{sim:d}/add_strategy.json'.format(sim=self['id']),
            data={'role': role, 'strategy': strategy})
        resp.raise_for_status()

    def add_dict(self, role_strat_dict):
        """Adds all of the roles and strategies in a dictionary

        The dictionary should be of the form {role: [strategies]}."""
        for role, strategies in role_strat_dict.items():
            self.add_role(role)
            for strategy in strategies:
                self.add_strategy(role, strategy)

    @_requires_id
    def remove_strategy(self, role, strategy):
        """Removes a strategy from the simulator"""
        resp = self._api._request(
            'post',
            'simulators/{sim:d}/remove_strategy.json'.format(sim=self[
                'id']),
            data={'role': role, 'strategy': strategy})
        resp.raise_for_status()

    def remove_dict(self, role_strat_dict):
        """Removes all of the strategies in a dictionary

        The dictionary should be of the form {role: [strategies]}. Empty roles
        are not removed."""
        for role, strategies in role_strat_dict.items():
            for strategy in strategies:
                self.remove_strategy(role, strategy)

    @_requires_id
    def create_generic_scheduler(
            self, name, active, process_memory, size, time_per_observation,
            observations_per_simulation, nodes,
            default_observation_requirement, configuration):
        """Creates a generic scheduler and returns it

        name           - The name for the scheduler.
        active         - True or false, specifying whether the scheduler is
                         initially active.
        process_memory - The amount of memory in MB that your simulations need.
        size           - The number of players for the scheduler.
        time_per_observation - The time you require to take a single
                         observation.
        observations_per_simulation - The number of observations to take per
                         simulation run.
        nodes          - The number of nodes required to run one of your
                         simulations.
        default_observation_requirement - The number of observations to take
                         of a profile in the absence of a specific request.
        configuration  - A dictionary representation that sets all the
                         run-time parameters for this scheduler.

        """
        resp = self._api._request(
            'post',
            'generic_schedulers',
            data={'scheduler': {
                'simulator_id': self['id'],
                'name': name,
                'active': active,
                'process_memory': process_memory,
                'size': size,
                'time_per_observation': time_per_observation,
                'observations_per_simulation': observations_per_simulation,
                'nodes': nodes,
                'default_observation_requirement':
                    default_observation_requirement,
                'configuration': configuration
            }})
        resp.raise_for_status()
        return self._api.scheduler(json.loads(resp.text))


class Scheduler(dict):
    """Get information and modify EGTA Online Scheduler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api = self.pop('_api')

    def get_info(self, verbose=False):
        """Get a scheduler information

        If `id` is specified then this is really efficient, and works for all
        scheduler. Otherwise `nam` may be specified, but only generic
        schedulers will be found."""

        if 'id' in self:
            data = {'granularity': 'with_requirements'} if verbose else {}
            resp = self._api._request(
                'get',
                'schedulers/{sched_id}.json'.format(sched_id=self['id']),
                data)
            resp.raise_for_status()
            result = self._api.scheduler(json.loads(resp.text))
            if verbose:
                result['scheduling_requirements'] = [
                    self.profile(prof, id=prof['profile_id']) for prof
                    in result.get('scheduling_requirements', None) or ()]

        else:
            result = utils.only(
                sched for sched in self._api.get_generic_schedulers()
                if sched['name'] == self['name'])
            self['id'] = result['id']
            if verbose:
                result = self.get_info(verbose=True)

        return result

    @_requires_id
    def update(self, **kwargs):
        """Update the parameters of a given scheduler

        kwargs are any of the mandatory arguments for create_generic_scheduler

        """
        resp = self._api._request(
            'put',
            'generic_schedulers/{sid:d}.json'.format(sid=self['id']),
            data={'scheduler': kwargs})
        resp.raise_for_status()

    @_requires_id
    def add_role(self, role, count):
        """Add a role with specific count to the scheduler"""
        resp = self._api._request(
            'post',
            'generic_schedulers/{sid:d}/add_role.json'.format(sid=self[
                'id']),
            data={'role': role, 'count': count})
        resp.raise_for_status()

    @_requires_id
    def remove_role(self, role):
        """Remove a role from the scheduler"""
        resp = self._api._request(
            'post',
            'generic_schedulers/{sid:d}/remove_role.json'.format(sid=self[
                'id']),
            data={'role': role})
        resp.raise_for_status()

    @_requires_id
    def delete_scheduler(self):
        """Delete a generic scheduler"""
        resp = self._api._request(
            'delete',
            'generic_schedulers/{sid:d}.json'.format(sid=self['id']))
        resp.raise_for_status()

    @_requires_id
    def profile(self, *args, **kwargs):
        """Get a profile object capable of profile manipulation

        Passing `id` will ensure fastest access, but `symmetry_groups` or
        `profile` will also work."""
        return self._api.profile(*args, scheduler_id=self['id'], **kwargs)

    def remove_all_profiles(self):
        """Removes all profiles from a scheduler"""
        for profile in self.get_info(verbose=True)['scheduling_requirements']:
            profile.remove()

    def num_running_profiles(self):
        """Get the number of currently running profiles"""
        update = self.get_info(verbose=True)
        # Sum profiles that aren't complete
        return sum(req['current_count'] < req['requirement']
                   for req in update['scheduling_requirements'] or ())

    def running_profiles(self):
        """Get a set of the active profiles"""
        update = self.get_info(verbose=True)
        return (req for req in update['scheduling_requirements'] or ()
                if req['current_count'] < req['requirement'])

    def are_profiles_still_active(self, profile_ids):
        """Returns true if any of the profile ids in profiles are active"""
        profile_ids = set(profile_ids)
        update = self.get_info(verbose=True)
        return any(
            req['profile_id'] in profile_ids
            and req['current_count'] < req['requirement']
            for req in update['scheduling_requirements'] or ())


class Profile(dict):
    """Class for manipulating profiles

    Key fields are `scheduler_id` the id of the scheduler these methods will
    manipulate. `id` the EGTA id of the profile. `symmetry_groups` the symmetry
    groups of the profile. `profile` the {role: {strategy: count}}
    representation of the profile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api = self.pop('_api')

    def get_info(self):
        """Gets information about the profile

        This is extremely slow if `id` is not present. This will only return a
        result if a profile with identical symmetry groups or profile exists in
        the scheduler matching `scheduler_id`."""
        if 'id' in self:
            resp = self._api._request(
                'get',
                'profiles/{pid:d}.json'.format(pid=self['id']))
            resp.raise_for_status()
            result = json.loads(resp.text)
            if 'scheduler_id' in self:
                result['scheduler_id'] = self['scheduler_id']
            return self._api.profile(result)

        elif 'scheduler_id' in self and (
                'profile' in self or 'symmetry_groups' in self):
            profile_desc = self.get_game_analysis_profile()
            for prof in (self._api.scheduler(id=self['scheduler_id'])
                         .get(verbose=True)['scheduling_requirements']):
                other_desc = prof.get_info().get_game_analysis_profile()
                if profile_desc == other_desc:
                    self['id'] = prof['id']
                    return prof

            raise ValueError('Could not find matching profile in scheduler')

        else:
            raise ValueError('Profile did not have either an id or a '
                             'scheduler_id and appropriate description')

    def get_game_analysis_profile(self):
        """Get a Game Analysis Profile object"""

        if 'profile' in self:
            return profile.Profile(self['profile'])
        elif 'symmetry_groups' in self:
            return profile.Profile.from_symmetry_groups(
                self['symmetry_groups'])
        elif 'id' in self:
            return self.get_info().get_game_analysis_profile()
        else:
            raise ValueError('Not enough info get get profile')

    def add(self, count):
        """Adds a profile with a given count to the scheduler

        If a profile with the same symmetry groups is already scheduled then
        this will have no effect.

        Setting update to True will cause a full scan through the profiles and
        remove the one that matches this one first. Only useful if updating the
        requested count of a profile.

        returns information about the added profile."""

        profile_desc = self.get_game_analysis_profile()

        resp = self._api._request(
            'post',
            'generic_schedulers/{sid:d}/add_profile.json'.format(
                sid=self['scheduler_id']),
            data={
                'assignment': str(profile_desc),
                'count': count
            })
        resp.raise_for_status()
        result = self._api.profile(json.loads(resp.text))
        if 'scheduler_id' in self:
            result['scheduler_id'] = self['scheduler_id']
        return result

    def update_count(self, count):
        """Update the count of a profile object"""
        prof_desc = self.get_game_analysis_profile()
        self.remove()
        self._api.profile(profile=prof_desc, scheduler_id=self[
                          'scheduler_id']).add(count)

    @_requires_id
    def remove(self):
        """Removes a profile from a scheduler

        If `id` is present this is fast, otherwise this is very slow."""
        resp = self._api._request(
            'post',
            'generic_schedulers/{sid:d}/remove_profile.json'.format(
                sid=self['scheduler_id']),
            data={'profile_id': self['id']})
        resp.raise_for_status()


class Game(dict):
    """Get information and manipulate EGTA Online Games"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api = self.pop('_api')

    def get_info(self, granularity='structure'):
        """Gets game information from EGTA Online

        granularity can be one of:

        structure    - returns the game information but no profile information.
                       (default)
        summary      - returns the game information and profiles with
                       aggregated payoffs.
        observations - returns the game information and profiles with data
                       aggregated at the observation level.
        full         - returns the game information and profiles with complete
                       observation information
        """
        if 'id' in self:
            # This call breaks convention because the api is broken, so we use
            # a different api.
            resp = self._api._non_api_request(
                'get',
                'games/{gid:d}.json'.format(gid=self['id']),
                data={'granularity': granularity})
            resp.raise_for_status()
            result = (
                json.loads(json.loads(resp.text))
                if granularity == 'structure'
                else json.loads(resp.text))

        else:
            result = utils.only(g for g in self._api.get_games() if g[
                                'name'] == self['name'])
            self['id'] = result['id']
            if granularity != 'structure':
                result = self.get_info(granularity=granularity)

        return result

    @_requires_id
    def add_role(self, role, count):
        """Adds a role to the game"""
        resp = self._api._request(
            'post',
            'games/{game:d}/add_role.json'.format(game=self['id']),
            data={'role': role, 'count': count})
        resp.raise_for_status()

    @_requires_id
    def remove_role(self, role):
        """Removes a role from the game"""
        resp = self._api._request(
            'post',
            'games/{game:d}/remove_role.json'.format(game=self['id']),
            data={'role': role})
        resp.raise_for_status()

    @_requires_id
    def add_strategy(self, role, strategy):
        """Adds a strategy to the game"""
        resp = self._api._request(
            'post',
            'games/{game:d}/add_strategy.json'.format(game=self['id']),
            data={'role': role, 'strategy': strategy})
        resp.raise_for_status()

    def add_dict(self, role_strat_dict):
        """Attempts to add all of the strategies in a dictionary

        The dictionary should be of the form {role: [strategies]}."""
        for role, strategies in role_strat_dict.items():
            for strategy in strategies:
                self.add_strategy(role, strategy)

    @_requires_id
    def remove_strategy(self, role, strategy):
        """Removes a strategy from the game"""
        resp = self._api._request(
            'post',
            'games/{game:d}/remove_strategy.json'.format(game=self['id']),
            data={'role': role, 'strategy': strategy})
        resp.raise_for_status()

    def remove_dict(self, role_strat_dict):
        """Removes all of the strategies in a dictionary

        The dictionary should be of the form {role: [strategies]}. Empty roles
        are not removed."""
        for role, strategies in role_strat_dict.items():
            for strategy in strategies:
                self.remove_strategy(role, strategy)
