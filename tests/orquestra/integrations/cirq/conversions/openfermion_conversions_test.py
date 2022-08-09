import pytest
from openfermion import IsingOperator, QubitOperator
from openfermion.testing import random_qubit_operator
from orquestra.quantum.operators import PauliSum, PauliTerm

from orquestra.integrations.cirq.conversions import from_openfermion, to_openfermion


class TestOpenFermionConversions:
    @pytest.mark.parametrize(
        "pauli_op",
        [PauliTerm.identity() * (4 + 0.5j), PauliSum.identity() * (4 + 0.5j)],
    )
    def test_to_openfermion_identity_operator(self, pauli_op):
        assert to_openfermion(pauli_op, QubitOperator) == QubitOperator("", 4 + 0.5j)
        assert to_openfermion(pauli_op, IsingOperator) == IsingOperator("", 4 + 0.5j)

    @pytest.mark.parametrize(
        "openfermion_op", [QubitOperator("", 4 + 0.5j), IsingOperator("", 4 + 0.5j)]
    )
    def test_from_openfermion_identity_operator(self, openfermion_op):
        pauli_op = from_openfermion(openfermion_op)
        assert pauli_op == PauliTerm.identity() * (4 + 0.5j)

    @pytest.mark.parametrize("pauli_op", [PauliTerm.identity() * 0, PauliSum()])
    def test_to_openfermion_zero(self, pauli_op):
        assert to_openfermion(pauli_op, QubitOperator) == QubitOperator()
        assert to_openfermion(pauli_op, IsingOperator) == IsingOperator()

    @pytest.mark.parametrize("openfermion_op", [QubitOperator(), IsingOperator()])
    def test_from_openfermion_zero(self, openfermion_op):
        pauli_op = from_openfermion(openfermion_op)
        assert pauli_op == PauliSum()

    @pytest.mark.parametrize(
        "pauli_op", [PauliTerm("X0*Y2*Z3", 9 + 0.5j), PauliSum("(9+0.5j)*X0*Y2*Z3")]
    )
    def test_to_openfermion_single_term(self, pauli_op):
        openfermion_op = to_openfermion(pauli_op)
        assert openfermion_op == QubitOperator("X0 Y2 Z3", 9 + 0.5j)

    @pytest.mark.parametrize(
        "pauli_op", [PauliTerm("Z0*Z2*Z3", 1 + 3.5j), PauliSum("(1+3.5j)*Z0*Z2*Z3")]
    )
    def test_to_openfermion_ising_single_term(self, pauli_op):
        openfermion_op = to_openfermion(pauli_op, IsingOperator)
        assert openfermion_op == IsingOperator("Z0 Z2 Z3", 1 + 3.5j)

    def test_from_openfermion_single_term(self):
        openfermion_op = QubitOperator("X0 Y2 Z3", 9 + 0.5j)
        pauli_op = from_openfermion(openfermion_op)
        assert pauli_op == PauliTerm("X0*Y2*Z3", 9 + 0.5j)
        ising_op = IsingOperator("Z0 Z2 Z3", 2j)
        assert from_openfermion(ising_op) == PauliTerm("(2j)*Z0*Z2*Z3")

    def test_multi_term(self):
        pauli_op = PauliSum("2*Z0*X5*Y7 + 8*Z5")
        openfermion_op = QubitOperator("2 [Z0 X5 Y7] + 8 [Z5]")
        assert to_openfermion(pauli_op) == openfermion_op
        assert from_openfermion(openfermion_op) == pauli_op

        ising_op = PauliSum("2*Z0*Z5*Z7 + 8*Z5")
        openfermion_ising_op = IsingOperator("2 [Z0 Z5 Z7] + 8 [Z5]")
        assert to_openfermion(ising_op, IsingOperator) == openfermion_ising_op
        assert from_openfermion(openfermion_ising_op) == ising_op

    def test_random_qubit_op(self):
        qubit_op = random_qubit_operator()
        transformed_op = from_openfermion(qubit_op)
        retransformed_op = to_openfermion(transformed_op)
        assert qubit_op == retransformed_op
