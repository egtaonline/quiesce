Quiesce
=======

A command line and python api for accessing egtaonline.
Also includes a script to automatically "quiesce" an empirical game on egtaonline.

Setup
-----

We recommend you install Quiesce in it's own virtual environment.
To use our recommended setup simply execute the following commands in the directory you want to store Quiesce in.

```
curl https://raw.githubusercontent.com/egtaonline/quiesce/master/quickuse_makefile > Makefile && make setup
```

`quiesce`, `egta`, and `watch` should now be accessible in the `bin` directory.
To update Quiesce, simply execute `make update` in the appropriate directory.


Cookbook
--------

Find the first simulation whose profile string matches the regex `<profile-regex>` and failed failed. Returns the error message.
```
bin/egta sims -r <profile-regex> | jq -r 'select(.state == "failed") | .folder' | head -n1 | xargs ./egta sims -f | jq -r .error_message
```

Find the first simulation whose profile string matches profile, and is finished. Returns whether it is complete or if it failed.
```
bin/egta sims -r <profile-regex> | jq -r 'select([.state == ("complete", "failed")] | any) | .state' | head -n1
```


Development
===========

`Makefile` has all of the relevant commands for settings up a development environment.
Typing `make` will print out everything it's setup to do.


TODO
----

* Simulator add by json results in duplicated strategies
