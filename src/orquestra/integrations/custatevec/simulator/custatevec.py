from ...cirq.simulator.qsim_simulator import QSimSimulator

import cirq

try:
    import qsimcirq  # type: ignore
except ModuleNotFoundError:
    warnings.warn(
        "qsimcirq is not imported. This library does not work with \n"
        "Python 3.10.0 or higher"
    )



class CustatevecSimulator(QSimSimulator):
        supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        noise_model=None,
        param_resolver: Optional[cirq.ParamResolverOrSimilarType] = None,
        qubit_order=cirq.ops.QubitOrder.DEFAULT,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        circuit_memoization_size: int = 0,
        qsim_options: Optional["qsimcirq.QSimOptions"] = None,
    ):

        qsim_options.use_gpu = True

        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsim_options,
            seed=seed,
            circuit_memoization_size=circuit_memoization_size,
        )

        super().__init__(simulator, noise_model, param_resolver, qubit_order)
