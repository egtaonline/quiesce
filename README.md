Empirical Game-Theoretic Analysis
=================================

[![Build Status](https://travis-ci.org/egtaonline/quiesce.svg?branch=master)](https://travis-ci.org/egtaonline/quiesce)

A command line tool for running egta on arbitrary simulators.

Setup
-----

FIXME

Usage
-----

You need a set of supplementary files to actually run this.
These files describe the game that's going to be run, and the processes of getting payoff data for each profile when requested.
This repository contains a sample simulator called `cdasim` that can be used for this purpose.
Below are some example uses

1. Perform the quiesce routine on a game that already has game data.
   One can also add noise to the payoffs for testing of equilibria procedures but this will just run it with no noise.

   ```
   egta --game-json cdasim/data_game.json quiesce game --load-game
   ```

2. Perform the quiesce routine on a game that's defined by a command line simulator.
   This will get profile data by sampling from the cdasim python simulator.

   ```
   egta --game-json cdasim/small_game.json quiesce sim -- python3 cdasim/sim.py 1 --single
   ```

   By default the quiesce routine only uses one payoff sample per profile.
   Setting `--count` to a larger number will help reduce the noise.

3. Perform the quiesce routine on a game with information on EGTA Online.
   The parameters specified here are for the same simulation that was uploaded there.

   ```
   egta --game-id 1466 quiesce --dpr buyers:2,sellers:2 egta --sim-memory 2048 --sim-time 60
   ```

   Note: To use this method, your EGTA Online credentials must be stored in a place where the python api can find them.
   Note: This game has already been solved, so this call will only fetch the initial game and then solve it without scheduling more profiles.


Development
===========

`Makefile` has all of the relevant commands for settings up a development environment.
Typing `make` will print out everything it's setup to do.

`make setup` will do a best effort to setup an appropriate development environment.
The script requires a python interpreter that's at least version 3.5.
To specify a different interpreter than the default lookup on your path use `make setup PYTHON=<alternate-python>`, e.g. on many ubuntu systems you might need to run `make setup PYTHON=python3`.
