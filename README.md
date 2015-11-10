Quiesce
=======

A script to automatically "quiesce" an empirical game on egtaonline

Setup
-----

```
$ git submodule init
$ git submodule update

$ sudo apt-get install python3 libatlas-base-dev gfortran
$ sudo pip3 install virtualenv

$ cd this/directory
$ virtualenv -p python3 .
$ . bin/activate

$ pip3 install -r game_analysis/requirements.txt -r requirements.txt
```

Use
---

The two main endpoints are `egta` and `quiesce`. `egta` is a general way to
access the egta api, while `quiesce` is a script for quiescing a game object.
