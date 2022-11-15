################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import Optional

import cirq

from ._base import CirqBasedSimulator


class CirqSimulator(CirqBasedSimulator):

    """Simulator using a cirq device (simulator or QPU).

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        seed: seed for random number generator.
        param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.

    Attributes:
        noise_model: an optional noise model to pass in for noisy simulations
        simulator: Cirq simulator this class uses.
        param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
    """

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model: cirq.NOISE_MODEL_LIKE = None,
        seed: Optional[int] = None,
        param_resolver: "cirq.ParamResolverOrSimilarType" = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
    ):
        simulator = cirq.Simulator(seed=seed)

        super().__init__(simulator, noise_model, param_resolver, qubit_order)
