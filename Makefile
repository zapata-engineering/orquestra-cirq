################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv ${VENV_NAME} && \
		${VENV_NAME}/bin/python3 -m pip install --upgrade pip && \
		${VENV_NAME}/bin/python3 -m pip install ./orquestra-quantum && \
		PYTHON="$({VENV_NAME}/bin/python3 -V )" && \
		${VENV_NAME}/bin/python3 -m pip install -e '.[dev]'

build-system-deps:
	$(PYTHON) -m pip install setuptools wheel "setuptools_scm>=6.0"


coverage:
	$(PYTHON) -m pytest -m "not custatevec" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests\
		--no-cov-on-fail \
		--cov-report xml \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!

totalcoverage:
	$(PYTHON) -m pytest -m "custatevec or not custatevec" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests\
		--no-cov-on-fail \
		--cov-report xml \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!