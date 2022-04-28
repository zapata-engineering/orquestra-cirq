# orquestra-cirq

An Orquestra Resource for Cirq

## Overview

`orquestra-cirq` is a Python module that exposes Cirq's simulators as an [`orquestra`](https://github.com/zapatacomputing/orquestra-quantum/blob/main/src/orquestra/quantum/api/backend.py) `QuantumSimulator`. It can be imported with:

```
from orquestra.integrations.cirq.simulator import CirqSimulator
```

In addition, it interfaces with the noise models and provides converters that allow switching between `cirq` circuits and those of `orquestra`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Cirq integration docs](http://docs.orquestra.io/other-resources/framework-integrations/cirq/).
For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).
