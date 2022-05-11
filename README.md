# orquestra-cirq

## What is it?

`orquestra-cirq` is a [Zapata](https://www.zapatacomputing.com) library holding modules for integrating cirq with [Orquestra](https://www.zapatacomputing.com/orquestra/).

## Installation

Even though it's intended to be used with Orquestra, `orquestra-cirq` can be also used as a Python module.
To install it, make to install `orquestra-quantum` first. Then you just need to run `pip install .` from the main directory.

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

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).
