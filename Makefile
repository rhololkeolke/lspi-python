.PHONY: html-docs clean clean-pyc test all flake8 sphinx-apidoc

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

clean: clean-pyc clean-docs
	
	
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

clean-docs:
	$(MAKE) -C docs clean

test:
	. lspienv/bin/activate; nosetests lspi_testsuite
