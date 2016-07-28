help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup - default setup necessary to use after checkout"
	@echo "ubuntu-setup - setup a clean installation on ubuntu (requires root)"

setup:
	virtualenv -p python3 .
	git submodule update --init
	bin/pip3 install -UI pip
	bin/pip3 install -r game_analysis/requirements.txt -r requirements.txt

ubuntu-install:
	sudo apt-get install python3 libatlas-base-dev gfortran libxml2-dev libxslt1-dev
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-install setup

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=game_analysis --exclude=Makefile --color=always

check:
	bin/flake8 egtaonline

format:
	bin/autopep8 -ri egtaonline

.PHONY: test
