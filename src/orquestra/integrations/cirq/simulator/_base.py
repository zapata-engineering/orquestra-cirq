################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import sys
from typing import List, Optional, Sequence, cast

import cirq
import numpy as np
from orquestra.quantum.api.wavefunction_simulator import BaseWavefunctionSimulator
from orquestra.quantum.circuits import Circuit, I
from orquestra.quantum.measurements import (
    ExpectationValues,
    Measurements,
    expectation_values_to_real,
)
from orquestra.quantum.operators import PauliRepresentation, get_sparse_operator
from orquestra.quantum.typing import StateVector

from ..conversions import export_to_cirq


class CirqBasedSimulator(BaseWavefunctionSimulator):
    """ABC for all Cirq based simulators."""

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        simulator,
        noise_model: cirq.NOISE_MODEL_LIKE = None,
        param_resolver: cirq.ParamResolverOrSimilarType = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
        normalize_wavefunction: bool = False,
    ):
        """initializes the parameters for the system or simulator

        Args:
            simulator: qsim or cirq simulator that is defined by the user
            noise_model: optional argument to define the noise model
            param_resolver: Optional arg that defines the parameters
            to run with the program.
            qubit_order: Optional arg that defines the ordering of qubits.
            normalize_wavefunction: whether to normalize the state vector after
                simulation of the quantum circuit, by default False.
        """
        super().__init__()
        self.noise_model = noise_model
        self.simulator = simulator
        self.param_resolver = param_resolver
        self.qubit_order = qubit_order
        self.normalize_wavefunction = normalize_wavefunction

    def _run_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:

        cirq_circuit = _prepare_measurable_cirq_circuit(circuit, self.noise_model)

        cirq_circuit.append(cirq.measure_each(*cirq_circuit.all_qubits()))

        result_object = self.simulator.run(
            cirq_circuit,
            param_resolver=self.param_resolver,
            repetitions=n_samples,
        )

        measurement = get_measurement_from_cirq_result_object(
            result_object, circuit.n_qubits, n_samples
        )

        return measurement

    def get_exact_noisy_expectation_values(
        self, circuit: Circuit, qubit_operator: PauliRepresentation
    ) -> ExpectationValues:
        """Compute exact expectation values w.r.t. given operator in presence of noise.

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
        if self.noise_model is None:
            raise RuntimeError(
                "Please provide noise model to get exact noisy expectation values"
            )

        cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))
        values = []

        for pauli_term in qubit_operator.terms:
            sparse_pauli_term_ndarray = get_sparse_operator(
                pauli_term, n_qubits=circuit.n_qubits
            ).toarray()
            if np.size(sparse_pauli_term_ndarray) == 1:
                expectation_value = sparse_pauli_term_ndarray[0][0]
                values.append(expectation_value)

            else:

                noisy_circuit = cirq_circuit.with_noise(self.noise_model)
                rho = self._extract_density_matrix(
                    self.simulator.simulate(noisy_circuit)
                )
                expectation_value = np.real(np.trace(rho @ sparse_pauli_term_ndarray))
                values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:

        cirq_circuit = _prepare_measurable_cirq_circuit(circuit, self.noise_model)

        initial_state = np.array(initial_state, np.complex64)

        simulated_result = self.simulator.simulate(
            cirq_circuit,
            param_resolver=self.param_resolver,
            qubit_order=self.qubit_order,
            initial_state=initial_state,
        )
        final_state_vector = np.asarray(simulated_result.final_state_vector)
        if self.normalize_wavefunction:
            final_state_vector = final_state_vector.astype(np.complex256)
            final_state_vector /= np.linalg.norm(final_state_vector)
        return final_state_vector

    def _extract_density_matrix(self, result):
        return result.density_matrix_of()


def _prepare_measurable_cirq_circuit(circuit, noise_model):
    """
    Export circuit to Cirq and add terminal measurements.

    Args:
        circuit (orquestra.quantum.circuit.Circuit): the circuit to prepare the state.
        noise_model: model to create a noisy circuit

    Returns:
        circuit to run on a cirq or qsim simulator
    """

    for i in range(circuit.n_qubits):
        circuit += I(i)

    cirq_circuit = cast(cirq.Circuit, export_to_cirq(circuit))

    if noise_model is not None:
        cirq_circuit = cirq_circuit.with_noise(noise_model)

    return cirq_circuit


def get_measurement_from_cirq_result_object(
    result_object: cirq.Result, n_qubits: int, n_samples: int
) -> Measurements:
    """Extract measurement bitstrings from cirq result object.
    Args:
        result_object: object returned by Cirq simulator's run or run_batch.
        n_qubits: number of qubits in full circuit (before exporting to cirq).
        n_samples: number of measured samples
    Return:
        Measurements.
    """
    keys = _find_reverse_permutation(result_object.data.columns, n_qubits)

    samples = list(
        tuple(measurement) for measurement in result_object.data.to_numpy()[:, keys]
    )

    measurement = Measurements(samples)
    return measurement


def _find_reverse_permutation(permutation, n_qubits):
    keys_to_indices = {f"q({n})": n for n in range(n_qubits)}
    result = [0 for _ in range(n_qubits)]
    for i, key in enumerate(permutation):
        result[keys_to_indices[key]] = i
    return result
