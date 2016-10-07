help:
	@echo "usage: make <tag>"
	@echo
	@echo "update - update quiesce & default setup necessary to use after checkout"
	@echo "ubuntu-reqs - install required files on ubuntu (requires root)"
	@echo "todo   - check for todo flags"
	@echo "check  - check for comformance to pep8 standards"
	@echo "format - autoformat python files"

update:
	git pull
	git submodule update --init
	pyvenv .
	bin/pip3 install -U pip
	bin/pip3 install -r game_analysis/requirements.txt -r requirements.txt

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv zlib1g-dev

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=game_analysis --exclude=Makefile --color=always

check:
	bin/flake8 egtaonline

format:
	bin/autopep8 -ri egtaonline
