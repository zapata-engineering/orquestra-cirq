################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import Dict, Union, cast

import cirq
import numpy as np
import qsimcirq
from orquestra.quantum.api.backend import StateVector
from orquestra.quantum.circuits import Circuit

from ..conversions import export_to_cirq
from ._base import CirqBaseSimulator


class QSimSimulator(CirqBaseSimulator):

    """Simulator using a qsimcirq simular which is optimized for GPU
    using cuStateVec (https://docs.nvidia.com/cuda/cuquantum/custatevec/index.html).
    Visit https://quantumai.google/qsim to learn more about qsimcirq.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        seed: seed for random number generator.
        param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
        circuit_memoization_size: Optional arg tht defines the number of
        last translated circuits to be memoized from simulation executions,
        to eliminate translation overhead.

    Attributes:
        noise_model: an optional noise model to pass in for noisy simulations
        qubit_order: qubit_order: Optional arg that defines the ordering of qubits.
        param_resolver: param_resolver: Optional arg that defines the
        parameters to run with the program.
    """

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model=None,
        seed=None,
        param_resolver=None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
        circuit_memoization_size: int = 0,
        qsim_options: Union[None, Dict, qsimcirq.QSimOptions] = None,
    ):
        self.qubit_order = qubit_order
        self.param_resolver = param_resolver

        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

        super().__init__(simulator, noise_model)

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        """Return the state vector at the end of the computation
        Args:
            circuit (Circuit): the circuit to prepare the state
            initial_state (StateVector): initial state of the system

        Returns:
            StateVector: Returns the final state of the computation as ndarray

        Raises:
            TypeError if this simulator is not QSimSimulator.
        """
        cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))

        initial_state = np.array(initial_state, np.complex64)

        simulated_result = self.simulator.simulate(
            cirq_circuit,
            param_resolver=self.param_resolver,
            qubit_order=self.qubit_order,
            initial_state=initial_state,
        )

        return simulated_result.final_state_vector

    def _extract_density_matrix(self, result):
        return result.density_matrix_of()
