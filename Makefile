PYTEST_ARGS =
FILES = egta test setup.py
PYTHON = python

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup  - setup for development"
	@echo "ubuntu-reqs - install required files on ubuntu (requires root)"
	@echo "todo   - check for todo flags"
	@echo "check  - check for comformance to pep8 standards"
	@echo "format - autoformat python files"
	@echo "test   - run tests with coverage"

setup:
	$(PYTHON) -m venv .
	bin/pip install -U pip setuptools
	bin/pip install -r requirements.txt -e .

test-all: PYTEST_ARGS += -m ''
test-all: test

test:
	bin/pytest test $(PYTEST_ARGS) --cov egta --cov test 2>/dev/null

ubuntu-reqs:
	sudo apt-get install libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv zlib1g-dev

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=lib64 --exclude=Makefile --color=always

check:
	bin/flake8 $(FILES)

format:
	bin/autopep8 -ri $(FILES)

upload:
	rm -rf dist
	cp ~/.pypirc ~/.pypirc.bak~ || touch ~/.pypirc.bak~
	echo '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nusername: strategic.reasoning.group' > ~/.pypirc
	bin/python setup.py sdist bdist_wheel && bin/twine upload dist/*; mv ~/.pypirc.bak~ ~/.pypirc

clean:
	rm -rf bin include lib lib64 man share pyvenv.cfg dist egta.egg-info

.PHONY: test clean format check todo help
