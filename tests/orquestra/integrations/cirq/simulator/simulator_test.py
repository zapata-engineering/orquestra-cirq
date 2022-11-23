################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from cirq import depolarize
from orquestra.quantum.api.circuit_runner_contracts import (
    CIRCUIT_RUNNER_CONTRACTS,
    STRICT_CIRCUIT_RUNNER_CONTRACTS,
)
from orquestra.quantum.api.wavefunction_simulator_contracts import (
    simulator_contracts_for_tolerance,
    simulator_contracts_with_nontrivial_initial_state,
    simulator_gate_compatibility_contracts,
)
from orquestra.quantum.circuits import CNOT, Circuit, H, X
from orquestra.quantum.operators import PauliSum

from orquestra.integrations.cirq.simulator import CirqSimulator, QSimSimulator


@pytest.fixture(
    params=[
        {
            "simulator": CirqSimulator,
            "atol_wavefunction": 1e-8,
        },
        {
            "simulator": QSimSimulator,
            "atol_wavefunction": 5e-7,
        },
    ],
)
def wf_simulator(request):
    return request.param


class TestCirqBasedSimulator:
    @pytest.fixture(autouse=True)
    def _request_simulator(self, wf_simulator):
        self.simulator = wf_simulator.get("simulator")
        self.atol_wavefunction = wf_simulator.get("atol_wavefunction")

    def test_setup_basic_simulators(self):
        simulator = self.simulator()
        if type(simulator).__name__ == "CirqSimulator":
            assert isinstance(simulator, CirqSimulator)
        if type(self.simulator).__name__ == "QSimSimulator":
            assert isinstance(simulator, QSimSimulator)
        assert simulator.noise_model is None

    def test_run_and_measure(self):
        # Given
        simulator = self.simulator()
        circuit = Circuit([X(0), CNOT(1, 2)])
        measurements = simulator.run_and_measure(circuit, n_samples=100)
        assert len(measurements.bitstrings) == 100

        for measurement in measurements.bitstrings:
            assert measurement == (1, 0, 0)

    def test_measuring_inactive_qubits(self):

        simulator = self.simulator()
        # Given
        circuit = Circuit([X(0), CNOT(1, 2)], n_qubits=4)

        measurements = simulator.run_and_measure(circuit, n_samples=100)
        assert len(measurements.bitstrings) == 100

        for measurement in measurements.bitstrings:
            assert measurement == (1, 0, 0, 0)

    def test_run_batch_and_measure(self):

        simulator = self.simulator()
        # Given
        circuit = Circuit([X(0), CNOT(1, 2)])
        n_circuits = 5
        n_samples = 100
        # When
        measurements_set = simulator.run_batch_and_measure(
            [circuit] * n_circuits, n_samples=[100] * n_circuits
        )
        # Then
        assert len(measurements_set) == n_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples
            for measurement in measurements.bitstrings:
                assert measurement == (1, 0, 0)

    def test_run_circuit_and_measure_seed(self):
        # Given
        circuit = Circuit([X(0), CNOT(1, 2)])
        simulator1 = self.simulator(seed=12)
        simulator2 = self.simulator(seed=12)

        # When
        measurements1 = simulator1.run_and_measure(circuit, n_samples=1000)
        measurements2 = simulator2.run_and_measure(circuit, n_samples=1000)

        # Then
        for (meas1, meas2) in zip(measurements1.bitstrings, measurements2.bitstrings):
            assert meas1 == meas2

    def test_get_wavefunction(self):
        simulator = self.simulator()
        # Given
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])

        # When
        wavefunction = simulator.get_wavefunction(circuit)
        # Then
        assert isinstance(wavefunction.amplitudes, np.ndarray)
        assert len(wavefunction.amplitudes) == 8
        assert np.isclose(
            wavefunction.amplitudes[0], (1 / np.sqrt(2) + 0j), atol=10e-15
        )
        assert np.isclose(
            wavefunction.amplitudes[7], (1 / np.sqrt(2) + 0j), atol=10e-15
        )

    def test_get_noisy_exact_expectation_values(self):
        # Given
        noise = 0.0002
        noise_model = depolarize(p=noise)
        simulator = self.simulator(noise_model=noise_model)
        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        qubit_operator = PauliSum("-1*Z0*Z1 + X0*X2")
        target_values = np.array([-0.9986673775881747, 0.0])

        expectation_values = simulator.get_exact_noisy_expectation_values(
            circuit, qubit_operator
        )
        np.testing.assert_almost_equal(
            expectation_values.values[0], target_values[0], 2
        )
        np.testing.assert_almost_equal(expectation_values.values[1], target_values[1])

    def test_normalization(self):
        simulator = self.simulator
        if isinstance(simulator(), CirqSimulator):
            pytest.skip("Normalization not required for CirqSimulator")
        # Given
        simulator1 = simulator(normalize_wavefunction=False)
        simulator2 = simulator(normalize_wavefunction=True)

        circuit = Circuit([H(0), CNOT(0, 1), CNOT(1, 2)])
        # When
        wavefunction = simulator1.get_wavefunction(circuit)
        normalized_wavefunction = simulator2.get_wavefunction(circuit)
        precision_error_without_normalization = abs(
            np.sum(np.abs(wavefunction.amplitudes) ** 2) - 1.0
        )
        precision_error_with_normalization = abs(
            np.sum(np.abs(normalized_wavefunction.amplitudes) ** 2) - 1.0
        )
        # Then
        assert np.isclose(
            precision_error_without_normalization,
            0.0,
            atol=self.atol_wavefunction,
        )
        assert np.isclose(precision_error_with_normalization, 0.0, atol=1e-15)


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_cirq_runner_fulfills_circuit_runner_contracts(wf_simulator, contract):
    simulator = wf_simulator.get("simulator")
    assert contract(simulator())


@pytest.mark.parametrize(
    "contract",
    simulator_contracts_for_tolerance()
    + simulator_contracts_with_nontrivial_initial_state(),
)
def test_cirq_wf_simulator_fulfills_wf_simulator_contracts(wf_simulator, contract):
    simulator = wf_simulator.get("simulator")
    assert contract(simulator())


@pytest.mark.parametrize("contract", STRICT_CIRCUIT_RUNNER_CONTRACTS)
def test_cirq_simulator_fulfills_strict_circuit_runnner(wf_simulator, contract):
    simulator = wf_simulator.get("simulator")
    assert contract(simulator())


@pytest.mark.parametrize("contract", simulator_gate_compatibility_contracts())
def test_cirq_simulator_uses_correct_gate_definitionscontract(wf_simulator, contract):
    simulator = wf_simulator.get("simulator")
    assert contract(simulator())
