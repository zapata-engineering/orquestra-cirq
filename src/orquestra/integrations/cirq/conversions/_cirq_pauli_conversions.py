################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import List, Optional, Union

import cirq
from orquestra.quantum.operators import PauliRepresentation


def pauliop_to_cirq_paulisum(
    pauli_operator: PauliRepresentation,
    qubits: Optional[Union[List[cirq.GridQubit], List[cirq.LineQubit]]] = None,
) -> cirq.PauliSum:
    """Convert an orquestra PauliSum or PauliTerm to a cirq PauliSum

    Args:
        pauli_operator: The openfermion operator to convert
        qubits: The qubits the operator is applied to

    Returns:
        cirq.PauliSum
    """
    operator_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    if qubits is None:
        qubits = [cirq.GridQubit(i, 0) for i in range(pauli_operator.n_qubits)]

    converted_sum = cirq.PauliSum()
    for term in pauli_operator.terms:

        # Identity term
        if term.is_constant:
            converted_sum += term.coefficient
            continue

        cirq_term: cirq.PauliString = cirq.PauliString()
        for qubit_index, operator in term.operations:
            cirq_term *= operator_map[operator](qubits[qubit_index])
        converted_sum += cirq_term * term.coefficient

    return converted_sum
