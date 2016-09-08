help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup - default setup necessary to use after checkout"
	@echo "ubuntu-setup - setup a clean installation on ubuntu (requires root)"

setup:
	git submodule update --init
	pyvenv .
	bin/pip3 install -U pip
	bin/pip3 install -r game_analysis/requirements.txt -r requirements.txt

pull:
	git pull

update: pull setup

ubuntu-install:
	sudo apt-get install python3 libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv zlib1g-dev

ubuntu-setup: ubuntu-install setup

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=game_analysis --exclude=Makefile --color=always

check:
	bin/flake8 egtaonline

format:
	bin/autopep8 -ri egtaonline

.PHONY: test
