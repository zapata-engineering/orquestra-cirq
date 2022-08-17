################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from openfermion import QubitOperator, SymbolicOperator  # type: ignore
from orquestra.quantum.operators import PauliRepresentation, PauliSum, PauliTerm


def from_openfermion(op: SymbolicOperator) -> PauliSum:
    """Convert from OpenFermion symbolic operator to Orquestra native PauliSum"""
    return PauliSum(
        [
            PauliTerm({idx: op for idx, op in term}, coefficient)
            for term, coefficient in op.terms.items()
        ]
    )


def to_openfermion(op: PauliRepresentation, operatorType=QubitOperator):
    """Convert from Orquestra native PauliSum or PauliTerm to OpenFermion symbolic
    operator

    Args:
        op: PauliSum to convert
        operatorType: a subclass of OpenFermion SymbolicOperator (can be OpenFermion
            QubitOperator or IsingOperator)
    """
    openfermion_op = operatorType()
    for term in op.terms:
        openfermion_op += operatorType(tuple(term.operations), term.coefficient)
    return openfermion_op
