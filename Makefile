.PHONY: html-docs clean clean-pyc clean-tests clean-releases test all flake8 sphinx-apidoc travis-test release upload-release

all: lspienv flake8 test html-docs 

lspienv: lspienv/bin/activate

lspienv/bin/activate: requirements.txt
	test -d lspienv || virtualenv lspienv
	. lspienv/bin/activate; pip install -Ur requirements.txt
	touch lspienv/bin/activate

flake8:
	. lspienv/bin/activate; flake8 lspi

sphinx-apidoc:
	. lspienv/bin/activate; sphinx-apidoc -f -e -o docs/source/autodoc lspi

html-docs: sphinx-apidoc
	. lspienv/bin/activate; PYTHONPATH=.. $(MAKE) -C docs html

clean: clean-pyc clean-docs clean-tests clean-releases


clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

clean-docs:
	$(MAKE) -C docs clean

clean-releases:
	rm -rf dist/
	rm -rf lspi_python.egg-info/
	rm -rf build/
	rm -f lspi-python-docs.zip

clean-tests:
	rm -rf htmlcov/

test:
	. lspienv/bin/activate; nosetests --config=setup.cfg lspi_testsuite

travis-test:
	nosetests --config=setup.cfg lspi_testsuite

release: lspienv flake8 test html-docs
	. lspienv/bin/activate; python setup.py sdist bdist_wheel
	zip -r lspi-python-docs.zip docs/build/html/*

upload-release: release
	. lspienv/bin/activate; twine upload -p $@ dist/*
