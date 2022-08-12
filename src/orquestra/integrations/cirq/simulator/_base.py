################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import sys
from typing import List, Sequence, cast

import cirq
import numpy as np
from orquestra.quantum.api.backend import QuantumSimulator, StateVector
from orquestra.quantum.circuits import Circuit, I
from orquestra.quantum.measurements import (
    ExpectationValues,
    Measurements,
    expectation_values_to_real,
)
from orquestra.quantum.operators import PauliRepresentation, get_sparse_operator

from ..conversions import export_to_cirq


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

    cirq_circuit = export_to_cirq(circuit)

    if noise_model is not None:
        cirq_circuit = cirq_circuit.with_noise(noise_model)

    cirq_circuit.append(cirq.measure_each(*cirq_circuit.all_qubits()))

    return cirq_circuit


class CirqBasedSimulator(QuantumSimulator):

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        simulator,
        noise_model: cirq.NOISE_MODEL_LIKE = None,
        param_resolver: cirq.ParamResolverOrSimilarType = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
    ):
        """initializes the parameters for the system or simulator

        Args:
            simulator: qsim or cirq simulator that is defined by the user
            noise_model: optional argument to define the noise model
            param_resolver: Optional arg that defines the parameters
            to run with the program.
            qubit_order: Optional arg that defines the ordering of qubits.
        """
        super().__init__()
        self.noise_model = noise_model
        self.simulator = simulator
        self.param_resolver = param_resolver
        self.qubit_order = qubit_order

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings.

        Args:
            circuit: the circuit to prepare the state.
            n_samples: number of bitstrings to measure. If None, `self.n_samples`
                is used.
        Returns:
            A list of bitstrings.
        """
        super().run_circuit_and_measure(circuit, n_samples)

        result_object = self.simulator.run(
            _prepare_measurable_cirq_circuit(circuit, self.noise_model),
            param_resolver=self.param_resolver,
            repetitions=n_samples,
        )

        measurement = get_measurement_from_cirq_result_object(
            result_object, circuit.n_qubits, n_samples
        )

        return measurement

    def run_circuitset_and_measure(
        self, circuitset: Sequence[Circuit], n_samples: Sequence[int]
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.
        Args:
            circuitset: a set of circuits to prepare the state.
            n_samples: number of bitstrings to measure. If None, `self.n_samples`
                is used. If an iterable, its-ith element corresponds to number
                of samples that will be taken from i-th circuit. If an int N,
                each circuit in circuitset will be measured N times.
        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        super().run_circuitset_and_measure(circuitset, n_samples)

        cirq_circuitset = [
            _prepare_measurable_cirq_circuit(circuit, self.noise_model)
            for circuit in circuitset
        ]

        simulation_result = self.simulator.run_batch(
            cirq_circuitset, repetitions=list(n_samples)
        )

        measurements_set = [
            get_measurement_from_cirq_result_object(
                sub_result[0], circuit.n_qubits, num_samples
            )
            for sub_result, circuit, num_samples in zip(
                simulation_result, circuitset, n_samples
            )
        ]

        return measurements_set

    def get_exact_expectation_values(
        self, circuit: Circuit, qubit_operator: PauliRepresentation
    ) -> ExpectationValues:
        """Compute exact expectation values with respect to given operator.

        Args:
            circuit: the circuit to prepare the state
            qubit_operator: the operator to measure
        Returns:
            the expectation values of each term in the operator
        """
        if self.noise_model is not None:
            return self.get_exact_noisy_expectation_values(circuit, qubit_operator)
        else:
            wavefunction = self.get_wavefunction(circuit).amplitudes

            # Pyquil does not support PauliSums with no terms.
            if len(qubit_operator.terms) == 0:
                return ExpectationValues(np.zeros((0,)))

            values = []

            for pauli_term in qubit_operator.terms:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=circuit.n_qubits
                ).toarray()
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    expectation_value = np.real(
                        wavefunction.conj().T @ sparse_pauli_term_ndarray @ wavefunction
                    )
                    values.append(expectation_value)

            return expectation_values_to_real(ExpectationValues(np.asarray(values)))

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
    numpy_samples = list(
        zip(
            *(
                result_object.measurements.get(f"q({sub_key})", [[0]] * n_samples)
                for sub_key in range(n_qubits)
            )
        )
    )

    samples = [
        tuple(int(key[0]) for key in numpy_bitstring)
        for numpy_bitstring in numpy_samples
    ]

    measurement = Measurements(samples)
    return measurement
