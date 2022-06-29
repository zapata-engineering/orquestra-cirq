################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv ${VENV_NAME} && \
		${VENV_NAME}/bin/python3 -m pip install --upgrade pip && \
		${VENV_NAME}/bin/python3 -m pip install ./orquestra-quantum && \
		PYTHON="$({VENV_NAME}/bin/python3 -V )" && \
		if [ "$PYTHON" == "Python 3.10.0"]; then ${VENV_NAME}/bin/python3 -m pip install -e '.[dev]'; else ${VENV_NAME}/bin/python3 -m pip install -e '.[dev, qsim]' ; fi

build-system-deps:
	$(PYTHON) -m pip install setuptools wheel "setuptools_scm>=6.0"
