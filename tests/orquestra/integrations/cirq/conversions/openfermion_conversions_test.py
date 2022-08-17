################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
from openfermion import IsingOperator, QubitOperator  # type: ignore
from openfermion.testing import random_qubit_operator  # type: ignore
from orquestra.quantum.operators import PauliSum, PauliTerm

from orquestra.integrations.cirq.conversions import from_openfermion, to_openfermion


class TestOpenFermionConversions:
    @pytest.mark.parametrize(
        "pauli_op",
        [PauliTerm.identity() * (4 + 0.5j), PauliSum.identity() * (4 + 0.5j)],
    )
    def test_identity_operator(self, pauli_op):
        openfermion_op = to_openfermion(pauli_op, QubitOperator)
        openfermion_ising_op = to_openfermion(pauli_op, IsingOperator)
        assert openfermion_op == QubitOperator("", 4 + 0.5j)
        assert openfermion_ising_op == IsingOperator("", 4 + 0.5j)

        back_to_pauli_op = from_openfermion(openfermion_op)
        back_to_pauli_op_from_ising = from_openfermion(openfermion_ising_op)
        assert back_to_pauli_op == pauli_op
        assert back_to_pauli_op_from_ising == pauli_op

    @pytest.mark.parametrize("pauli_op", [PauliTerm.identity() * 0, PauliSum()])
    def test_zero_operator(self, pauli_op):
        of_op = to_openfermion(pauli_op, QubitOperator)
        of_op_ising = to_openfermion(pauli_op, IsingOperator)
        assert of_op == QubitOperator()
        assert of_op_ising == IsingOperator()

        back_to_pauli_op = from_openfermion(of_op)
        back_to_pauli_op_from_ising = from_openfermion(of_op_ising)
        assert back_to_pauli_op == pauli_op
        assert back_to_pauli_op_from_ising == pauli_op

    @pytest.mark.parametrize(
        "pauli_op", [PauliTerm("X0*Y2*Z3", 9 + 0.5j), PauliSum("(9+0.5j)*X0*Y2*Z3")]
    )
    def test_single_term(self, pauli_op):
        openfermion_op = to_openfermion(pauli_op)
        assert openfermion_op == QubitOperator("X0 Y2 Z3", 9 + 0.5j)
        back_to_pauli_op = from_openfermion(openfermion_op)
        assert back_to_pauli_op == pauli_op

    @pytest.mark.parametrize(
        "pauli_op", [PauliTerm("Z0*Z2*Z3", 2j), PauliSum("(2j)*Z0*Z2*Z3")]
    )
    def test_ising_single_term(self, pauli_op):
        ising_op = to_openfermion(pauli_op, IsingOperator)
        assert ising_op == IsingOperator("Z0 Z2 Z3", 2j)
        assert from_openfermion(ising_op) == pauli_op

    def test_multi_term(self):
        pauli_op = PauliSum("2*Z0*X5*Y7 + 8*Z5")
        openfermion_op = QubitOperator("2 [Z0 X5 Y7] + 8 [Z5]")
        assert to_openfermion(pauli_op) == openfermion_op
        assert from_openfermion(openfermion_op) == pauli_op

    def test_ising_multi_term(self):
        ising_op = PauliSum("2*Z0*Z5*Z7 + 8*Z5")
        openfermion_ising_op = IsingOperator("2 [Z0 Z5 Z7] + 8 [Z5]")
        assert to_openfermion(ising_op, IsingOperator) == openfermion_ising_op
        assert from_openfermion(openfermion_ising_op) == ising_op

    def test_random_qubit_op(self):
        of_qubit_op = random_qubit_operator()
        pauli_op = from_openfermion(of_qubit_op)
        back_to_of_qubit_op = to_openfermion(pauli_op)
        assert of_qubit_op == back_to_of_qubit_op
