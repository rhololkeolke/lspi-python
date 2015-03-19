.PHONY: html-docs clean clean-pyc test all flake8

all: flake8 test html-docs
	
flake8:
	flake8 lspi

html-docs:
	@sphinx-apidoc -f -o docs/source/autodoc lspi
	PYTHONPATH=.. $(MAKE) -C docs html

clean: clean-pyc clean-docs
	
	
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

clean-docs:
	$(MAKE) -C docs clean

test:
	nosetests lspi_testsuite
