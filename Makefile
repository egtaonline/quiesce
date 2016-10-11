help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup  - setup for development"
	@echo "ubuntu-reqs - install required files on ubuntu (requires root)"
	@echo "todo   - check for todo flags"
	@echo "check  - check for comformance to pep8 standards"
	@echo "format - autoformat python files"

setup:
	pyvenv .
	bin/pip install -U pip setuptools
	bin/pip install --process-dependency-links -e .

ubuntu-reqs:
	sudo apt-get install libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv zlib1g-dev

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=game_analysis --exclude=Makefile --color=always

check:
	bin/flake8 egtaonline

format:
	bin/autopep8 -ri egtaonline

clean:
	rm -rf bin include lib lib64 man share pyvenv.cfg
