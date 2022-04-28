################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import List, Union

import cirq
from orquestra.quantum.openfermion import QubitOperator, count_qubits


def qubitop_to_paulisum(
    qubit_operator: QubitOperator,
    qubits: Union[List[cirq.GridQubit], List[cirq.LineQubit]] = None,
) -> cirq.PauliSum:
    """Convert and openfermion QubitOperator to a cirq PauliSum

    Args:
        qubit_operator (openfermion.QubitOperator): The openfermion operator to convert
        qubits()

    Returns:
        cirq.PauliSum
    """
    operator_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    if qubits is None:
        qubits = [cirq.GridQubit(i, 0) for i in range(count_qubits(qubit_operator))]

    converted_sum = cirq.PauliSum()
    for term, coefficient in qubit_operator.terms.items():

        # Identity term
        if len(term) == 0:
            converted_sum += coefficient
            continue

        cirq_term: cirq.PauliString = cirq.PauliString()
        for qubit_index, operator in term:
            cirq_term *= operator_map[operator](qubits[qubit_index])
        converted_sum += cirq_term * coefficient

    return converted_sum
