help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup - default setup necessary to use after checkout"
	@echo "ubuntu-setup - setup a clean installation on ubuntu (requires root)"

setup:
	virtualenv -p python3 .
	git submodule init
	git submodule update
	bin/pip3 install -U pip
	bin/pip3 install -r game_analysis/requirements.txt -r requirements.txt

ubuntu-install:
	sudo apt-get install python3 libatlas-base-dev gfortran
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-install setup

check:
	bin/flake8 egtaonline

.PHONY: test
