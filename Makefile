UNAME := $(shell uname -m)

clean:
	rm -rf coverage .coverage build dist

clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

doc:
	sphinx-build -b html doc doc-build

test:
	pytest eelbrain

test-no-gui:
	pytest eelbrain --no-gui

test-only-gui:
	pytest eelbrain/_wxgui eelbrain/plot eelbrain/tests/test_examples.py eelbrain/tests/test_fmtxt.py eelbrain/tests/test_report.py

test-gui-nomayavi:
	pytest --no-mayavi eelbrain/_wxgui eelbrain/plot eelbrain/tests/test_examples.py eelbrain/tests/test_fmtxt.py eelbrain/tests/test_report.py

pypi:
	rm -rf build dist
	python setup.py sdist bdist_wheel
	twine upload dist/*

.PHONY: clean clean-py doc testw pypi
