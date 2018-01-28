PYTEST_ARGS =
FILES = egta test setup.py
PYTHON = python3

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup    - setup for development"
	@echo "ubuntu-reqs - install required files on ubuntu (requires root)"
	@echo "todo     - check for todo flags"
	@echo "check    - check for comformance to pep8 standards"
	@echo "format   - autoformat python files"
	@echo "test     - run fast tests with coverage"
	@echo "test-all - run all tests with coverage"
	@echo "publish  - publish package to pypi"
	@echo "docs     - build documentation"
	@echo "clean    - remove build artifacts"
	@echo "travis   - run travis test script"

setup:
	$(PYTHON) -m venv .
	bin/pip install -U pip setuptools
	bin/pip install -e '.[dev]'

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

publish:
	bin/python setup.py sdist bdist_wheel
	bin/twine upload -u strategic.reasoning.group dist/*

docs:
	bin/python setup.py build_sphinx -b html

clean:
	rm -rf bin include lib lib64 man share pyvenv.cfg dist egta.egg-info

travis: PYTEST_ARGS += -v -n2
travis: check test

.PHONY: setup test-all test ubuntu-reqs todo check format publish clean travis docs
