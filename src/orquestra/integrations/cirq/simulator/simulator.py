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
from ._base import CirqBaseSimulator


class CirqSimulator(CirqBaseSimulator):

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

    def __init__(self, noise_model=None, seed: int = None):
        simulator = (
            cirq.DensityMatrixSimulator(dtype=np.complex128, seed=seed)
            if noise_model is not None
            else cirq.Simulator(seed=seed)
        )

        super().__init__(simulator, noise_model)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:

        cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))
        return cirq_circuit.final_state_vector(
            initial_state=cast(cirq.STATE_VECTOR_LIKE, initial_state)
        )

    def _extract_density_matrix(self, result):
        return result.final_density_matrix
