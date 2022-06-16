################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
from typing import Dict, List, Optional, Sequence, Union, cast

import cirq
import numpy as np
import qsimcirq
from orquestra.quantum.api.backend import QuantumSimulator, StateVector
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.measurements import (
    ExpectationValues,
    Measurements,
    expectation_values_to_real,
)
from orquestra.quantum.openfermion import SymbolicOperator, get_sparse_operator

from ..conversions import export_to_cirq
from .simulator import (
    CirqSimulator,
    _prepare_measurable_cirq_circuit,
    get_measurement_from_cirq_result_object,
)


# TODO: We need to see if this even make sense...
class QSimSimulator(CirqSimulator, QuantumSimulator):
    """Simulator using a cirq device (simulator or QPU).
    TODO
    Currently this Simulator uses cirq.Simulator if noise_model is None and
    cirq.DensityMatrixSimulator otherwise.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        TODO

    Attributes:
        noise_model: an optional noise model to pass in for noisy simulations
        simulator: Cirq simulator this class uses.
        TODO
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
        super().__init__()

        self.noise_model = noise_model
        self.qubit_order = qubit_order
        self.param_resolver = param_resolver

        self.simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            # noise=noise_model, #TODO: investigate later
            circuit_memoization_size=circuit_memoization_size,
        )

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:

        super().run_circuit_and_measure(circuit, n_samples)

        result_object = self.simulator.run(
            _prepare_measurable_cirq_circuit(
                circuit, self.noise_model, param_resolver=self.param_resolver
            ),
            repetitions=n_samples,
        )

        measurement = get_measurement_from_cirq_result_object(
            result_object, circuit.n_qubits, n_samples
        )

        return measurement

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))

        initial_state = np.array(initial_state, np.complex64)

        simulated_result = self.simulator.simulate(
            cirq_circuit,
            param_resolver=self.param_resolver,
            qubit_order=self.qubit_order,
            initial_state=initial_state,
        )

        return simulated_result.final_state_vector

    # Same as CirqSimulator
    def get_exact_noisy_expectation_values(
        self, circuit: Circuit, qubit_operator: SymbolicOperator
    ) -> ExpectationValues:
        """Compute exact expectation values w.r.t. given operator in presence of noise.
        TODO
        Note that this method can be used only if simulator's noise_model is not set
        to None.

        Args:
            circuit: the circuit to prepare the state
            qubit_operator: the operator to measure
        Returns:
            the expectation values of each term in the operator
        Raises:
            RuntimeError if this simulator's noise_model is None.
        """
        # TODO: see how to handle this in qsim

        if self.noise_model is None:
            raise RuntimeError(
                "Please provide noise model to get exact noisy expectation values"
            )
        else:
            cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))
            values = []

            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=circuit.n_qubits
                ).toarray()
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    noisy_circuit = cirq_circuit.with_noise(self.noise_model)
                    rho = self.simulator.simulate(noisy_circuit).final_density_matrix
                    expectation_value = np.real(
                        np.trace(
                            rho @ sparse_pauli_term_ndarray
                        )  # TODO: understand the background behind this
                    )
                    values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))


# class QsimhSimulator(QuantumSimulator):
#     pass
