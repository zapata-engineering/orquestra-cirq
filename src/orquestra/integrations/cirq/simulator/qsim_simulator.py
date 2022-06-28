################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import Dict, Optional, Union, cast

import cirq
import numpy as np
import qsimcirq
from orquestra.quantum.api.backend import StateVector
from orquestra.quantum.circuits import Circuit

from ..conversions import export_to_cirq
from ._base import CirqBasedSimulator


class QSimSimulator(CirqBasedSimulator):

    """Simulator using a qsimcirq simulator which is optimized for GPU
    using cuStateVec (https://docs.nvidia.com/cuda/cuquantum/custatevec/index.html).
    Visit https://quantumai.google/qsim to learn more about qsimcirq.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
        seed: seed for random number generator.
        circuit_memoization_size: Optional arg tht defines the number of
        last translated circuits to be memoized from simulation executions,
        to eliminate translation overhead.
        qsim_options:  An options dict or QSimOptions object with options
        to use for all circuits run using this simulator. See QSimOptions from
        qsimcirq for more details.

    Attributes:
        simulator: Qsim simulator this class uses with the options defined.
        noise_model: an optional noise model to pass in for noisy simulations
        qubit_order: qubit_order: Optional arg that defines the ordering of qubits.
        param_resolver: param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
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
        qsim_options: Optional[Union[Dict, qsimcirq.QSimOptions]] = None,
    ):

        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

        super().__init__(simulator, noise_model, param_resolver, qubit_order)
