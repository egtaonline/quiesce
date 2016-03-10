Quiesce
=======

A script to automatically "quiesce" an empirical game on egtaonline

Setup
-----

```
$ git submodule init
$ git submodule update

$ sudo apt-get install python3 libatlas-base-dev gfortran libxml2-dev libxslt1-dev
$ sudo pip3 install virtualenv

$ cd this/directory
$ virtualenv -p python3 .
$ . bin/activate

$ pip3 install -r game_analysis/requirements.txt -r requirements.txt
```

Cookbook
--------

Find the first simulation whose profile string matches the regex `<profile-regex>` and failed failed. Returns the error message.
```
./egta sims -r <profile-regex> | jq -r 'select(.state == "failed") | .folder' | head -n1 | xargs ./egta sims -f | jq -r .error_message
```

Find the first simulation whose profile string matches profile, and is finished. Returns whether it is complete or if it failed.
```
./egta sims -r <profile-regex> | jq -r 'select([.state == ("complete", "failed")] | any) | .state' | head -n1
```


TODO
----

* Simulator add by json results in duplicated strategies
* Make mixture set that uses LSH
* Use mixture set to short circuit equilibrium exploration
