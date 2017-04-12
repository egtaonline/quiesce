FILES = egta test setup.py

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup  - setup for development"
	@echo "ubuntu-reqs - install required files on ubuntu (requires root)"
	@echo "todo   - check for todo flags"
	@echo "check  - check for comformance to pep8 standards"
	@echo "format - autoformat python files"
	@echo
	@echo "add EGTA_TESTS=ON to run egta tests as well"

setup:
	pyvenv .
	bin/pip install -U pip setuptools
	bin/pip install -e .
	bin/pip install -r requirements.txt

test:
	bin/pytest test

coverage:
	bin/pytest test --cov egta --cov test 2>/dev/null

ubuntu-reqs:
	sudo apt-get install libatlas-base-dev gfortran libxml2-dev libxslt1-dev python3-venv zlib1g-dev

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-dir=lib64 --exclude=Makefile --color=always

check:
	bin/flake8 $(FILES)

format:
	bin/autopep8 -ri $(FILES)

upload:
	cp ~/.pypirc ~/.pypirc.bak~ || touch ~/.pypirc.bak~
	echo '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nrepository: https://pypi.python.org/pypi\nusername: strategic.reasoning.group' > ~/.pypirc
	bin/python setup.py sdist bdist_wheel upload; mv ~/.pypirc.bak~ ~/.pypirc

clean:
	rm -rf bin include lib lib64 man share pyvenv.cfg dist egta.egg-info

.PHONY: test coverage clean format check todo help
