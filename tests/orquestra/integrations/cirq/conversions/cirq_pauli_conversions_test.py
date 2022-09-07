################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import unittest

from cirq import GridQubit, LineQubit, PauliString
from cirq import PauliSum as CirqPauliSum
from cirq import X, Y, Z
from orquestra.quantum.operators import PauliTerm

from orquestra.integrations.cirq.conversions import pauliop_to_cirq_paulisum


class TestQubitOperator(unittest.TestCase):
    def test_pauliop_to_cirq_paulisum_identity_operator(self):
        # Given
        qubit_operator = PauliTerm.identity() * 4

        # When
        paulisum = pauliop_to_cirq_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, ())
        self.assertEqual(paulisum, CirqPauliSum() + 4)

    def test_pauliop_to_cirq_paulisum_z0z1_operator(self):
        # Given
        qubit_operator = PauliTerm("Z0*Z1", -1.5)
        expected_qubits = (GridQubit(0, 0), GridQubit(1, 0))
        expected_paulisum = (
            CirqPauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = pauliop_to_cirq_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_pauliop_to_cirq_paulisum_setting_qubits(self):
        # Given
        qubit_operator = PauliTerm("Z0*Z1", -1.5)
        expected_qubits = (LineQubit(0), LineQubit(5))
        expected_paulisum = (
            CirqPauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = pauliop_to_cirq_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_pauliop_to_cirq_paulisum_more_terms(self):
        # Given
        qubit_operator = (
            PauliTerm("Z0*Z1*Z2", -1.5) + PauliTerm("X0", 2.5) + PauliTerm("Y1", 3.5)
        )
        expected_qubits = (LineQubit(0), LineQubit(5), LineQubit(8))
        expected_paulisum = (
            CirqPauliSum()
            + (
                PauliString(Z.on(expected_qubits[0]))
                * PauliString(Z.on(expected_qubits[1]))
                * PauliString(Z.on(expected_qubits[2]))
                * -1.5
            )
            + (PauliString(X.on(expected_qubits[0]) * 2.5))
            + (PauliString(Y.on(expected_qubits[1]) * 3.5))
        )

        # When
        paulisum = pauliop_to_cirq_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)
