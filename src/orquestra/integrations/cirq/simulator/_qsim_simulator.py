################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import sys
import warnings
from typing import Optional

import cirq

try:
    import qsimcirq  # type: ignore
except ModuleNotFoundError:
    warnings.warn("qsimcirq is not imported")
from ._base import CirqBasedSimulator


class QSimSimulator(CirqBasedSimulator):

    """Integration with qsim simulator.
    In order to run on GPU using cuStateVec
    (https://docs.nvidia.com/cuda/cuquantum/custatevec/index.html)
    please provide `use_gpu=True` and `gpu_mode=1` in `qsim_options`.
    Visit https://quantumai.google/qsim to learn more about qsimcirq


    Args:
        noise_model: an optional noise model to pass in for noisy simulations
        param_resolver: Optional arg that defines the
        parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
        seed: seed for random number generator.
        circuit_memoization_size: Optional arg tht defines the number of
            last translated circuits to be memoized from simulation executions,
            to eliminate translation overhead.
        qsim_options:  An options dict or QSimOptions object with options
            to use for all circuits run using this simulator. See QSimOptions from
            qsimcirq for more details.
        normalize_wavefunction: Whether to normalize the state vector after
            simulation of the quantum circuit, by default False. This flag is
            exposed because sometimes, the resulting state vector from a qsim
            circuit simulation is not normalized up to machine precision, which
            can cause issues with some applications such as sampling using a
            probability vector.

    Attributes:
        simulator: Qsim simulator this class uses with the options defined.
        noise_model: an optional noise model to pass in for noisy simulations
        param_resolver: param_resolver: Optional arg that defines the
            parameters to run with the program.
        qubit_order: Optional arg that defines the ordering of qubits.
    """

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model: cirq.NOISE_MODEL_LIKE = None,
        param_resolver: Optional[cirq.ParamResolverOrSimilarType] = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        circuit_memoization_size: int = 0,
        qsim_options: Optional["qsimcirq.QSimOptions"] = None,
        normalize_wavefunction: bool = False,
    ):

        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

        super().__init__(
            simulator, noise_model, param_resolver, qubit_order, normalize_wavefunction
        )
