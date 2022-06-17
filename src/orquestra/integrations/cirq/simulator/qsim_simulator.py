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


class QSimSimulator(CirqSimulator, QuantumSimulator):

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
        super().__init__()

        self.noise_model = noise_model
        self.qubit_order = qubit_order
        self.param_resolver = param_resolver

        self.simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

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
            _prepare_measurable_cirq_circuit(
                circuit, self.noise_model, param_resolver=self.param_resolver
            ),
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

        result = self.simulator.run_batch(cirq_circuitset, repetitions=list(n_samples))

        measurements_set = [
            get_measurement_from_cirq_result_object(
                sub_result[0], circuit.n_qubits, num_samples
            )
            for sub_result, circuit, num_samples in zip(result, circuitset, n_samples)
        ]

        return measurements_set

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

        if isinstance(self.simulator, qsimcirq.QSimSimulator):
            simulated_result = self.simulator.simulate(
                cirq_circuit,
                param_resolver=self.param_resolver,
                qubit_order=self.qubit_order,
                initial_state=initial_state,
            )
        else:
            raise TypeError("Simulator is not QSimSimulator")

        return simulated_result.final_state_vector

    def get_exact_noisy_expectation_values(
        self, circuit: Circuit, qubit_operator: SymbolicOperator
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

        elif not isinstance(self.simulator, qsimcirq.QSimSimulator):
            raise TypeError("Simulator is not QSimSimulator")
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

                    rho = self.simulator.simulate(noisy_circuit).density_matrix_of()
                    expectation_value = np.real(
                        np.trace(rho @ sparse_pauli_term_ndarray)
                    )
                    values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))
