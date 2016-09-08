Quiesce
=======

A script to automatically "quiesce" an empirical game on egtaonline

Setup
-----

To run the quiesce script, you need to install the following dependencies on ubuntu.
Similar packages exist on mac and can be installed with homebrew.

```
$ sudo apt-get install python3 libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv
```

After that type

```
make update
```

and the quiesce script should be ready to use.


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
