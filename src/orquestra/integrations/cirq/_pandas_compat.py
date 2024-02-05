################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################
"""Workaround for pandas 2.2.0 raising warning related to pyarrow. """

import warnings


def preload_pandas_without_warnings():
    """Workaround for pandas 2.2.0 raising warning related to pyarrow.

    Run this function before importing a module which depends on pandas
    to avoid deprecation warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # The warning's content:
        #
        # DeprecationWarning:
        # Pyarrow will become a required dependency of pandas in the next majorrelease
        # of pandas (pandas 3.0),
        # (to allow more performant data types, such as the Arrow string type, and
        # better interoperability with other libraries)
        # but was not found to be installed on your system.
        # If this would cause problems for you,
        # please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
        import pandas
