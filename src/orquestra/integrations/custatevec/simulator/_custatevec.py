################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import sys
import warnings
from typing import Optional

import cirq

from ...cirq.simulator._qsim_simulator import QSimSimulator

try:
    import qsimcirq  # type: ignore
except ModuleNotFoundError:
    warnings.warn("qsimcirq is not imported")


class CuStateVecSimulator(QSimSimulator):
    """Qsimcirq simulator that using Nvidia GPUs for all simulations.

    CUDA toolkit and some dependency tools must be installed. The installation
    guidelines are provided in https://quantumai.google/qsim/tutorials/gcp_gpu.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        param_resolver: Optional arg that defines the parameters to run with
          the program.
        qubit_order: Optional arg that defines the ordering of qubits.
        seed: seed for random number generator.
        circuit_memoization_size: Optional arg tht defines the number of
          last translated circuits to be memoized from simulation executions,
          to eliminate translation overhead.
        qsim_options: An options dict or QSimOptions object with options
          to use for all circuits run using this simulator. See QSimOptions from
          qsimcirq for more details.
    """

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
            qsim_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
        else:
            qsim_options.use_gpu = True
            qsim_options.gpu_mode = 1

        super().__init__(
            noise_model=noise_model,
            param_resolver=param_resolver,
            qubit_order=qubit_order,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
            qsim_options=qsim_options,
        )
