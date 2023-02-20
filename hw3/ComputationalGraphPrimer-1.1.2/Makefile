# GNU -*- makefile -*-

#VERSION := ${shell python -c "import ComputationalGraphPrimer;print ComputationalGraphPrimer.__version__"}
VERSION := ${shell python3 -c "import os; os.chdir('ComputationalGraphPrimer'); import ComputationalGraphPrimer; print(ComputationalGraphPrimer.__version__)"}

default:
	@echo
	@echo "  *** Welcome to ComputationalGraphPrimer ${VERSION} ***"
	@echo
	@echo "  docs   -  Build documentation (html)"
	@echo "  help   -  Open the documentation"
	@echo
	@echo "  clean  -  Remove temporary files"
	@echo "  test   -  Run the unittests"
	@echo "  check  -  Look for rough spots"
	@echo "  sdist  -  Build a source distribution tar ball"

docs:
	pydoc3 -w ComputationalGraphPrimer/ComputationalGraphPrimer.py

help:
	open ComputationalGraphPrimer-${VERSION}.html

clean:
	rm -f *.pyc *~

real-clean: clean
	rm -f MANIFEST  *.html ComputationalGraphPrimer-py.info
	rm -rf build dist

# Run the unittest
test:
	@echo
	@echo Testing...
	@echo
	python3 ./TestComputationalGraphPrimer/Test.py 

sdist: test
	@echo
	@echo Building a source distribution...
	@echo
	./setup.py sdist --formats=gztar,bztar

# Look for rough spots
check:
	@grep -n FIX *.py *.in PKG-INFO Makefile | grep -v grep
	@echo
	pychecker ComputationalGraphPrimer
