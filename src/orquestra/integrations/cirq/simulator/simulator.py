################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import cast

import cirq
import numpy as np

# from cirq import ops
from orquestra.quantum.api.backend import StateVector
from orquestra.quantum.circuits import Circuit

from ..conversions import export_to_cirq
from ._base import CirqBasedSimulator


class CirqSimulator(CirqBasedSimulator):

    """Simulator using a cirq device (simulator or QPU).

    Currently this Simulator uses cirq.Simulator if noise_model is None and
    cirq.DensityMatrixSimulator otherwise.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        seed: seed for random number generator.

    Attributes:
        noise_model: an optional noise model to pass in for noisy simulations
        simulator: Cirq simulator this class uses.
    """

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model=None,
        seed: int = None,
        param_resolver: "cirq.ParamResolverOrSimilarType" = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
    ):
        simulator = cirq.Simulator(seed=seed)

        super().__init__(simulator, noise_model, param_resolver, qubit_order)
