################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import sys
import warnings
from typing import Optional

import cirq

from ...cirq.simulator._base import CirqBasedSimulator

try:
    import qsimcirq  # type: ignore
except ModuleNotFoundError:
    warnings.warn("qsimcirq is not imported")


class CustatevecSimulator(CirqBasedSimulator):
    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model=None,
        param_resolver: Optional[cirq.ParamResolverOrSimilarType] = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        circuit_memoization_size: int = 0,
        qsim_options: Optional["qsimcirq.QSimOptions"] = None,
    ):

        if qsim_options is None:
            qsim_options = qsimcirq.QSimOptions(use_gpu=True)
        else:
            qsim_options.use_gpu = True

        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

        super().__init__(simulator, noise_model, param_resolver, qubit_order)
