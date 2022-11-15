# orquestra-cirq

## What is it?

`orquestra-cirq` is a [Zapata](https://www.zapatacomputing.com) library holding modules for integrating cirq and qsimcirq with [Orquestra](https://www.zapatacomputing.com/orquestra/).

## Installation

Even though it's intended to be used with Orquestra, `orquestra-cirq` can be also used as a Python module.
To install it, make to install `orquestra-quantum` first. Then you just need to run `pip install .` from the main directory.
If you want to import `QSimSimulator`, you have to install it with the extra dependencies by running `pip install -e .[qsim]`.

## Overview

`orquestra-cirq` is a Python module that exposes Cirq's and qsim's simulators as an [`orquestra`](https://github.com/zapatacomputing/orquestra-quantum/blob/main/src/orquestra/quantum/api/backend.py) `QuantumSimulator`. They can be imported with:

```
from orquestra.integrations.cirq.simulator import CirqSimulator
from orquestra.integrations.cirq.simulator import QSimSimulator
```

In addition, it interfaces with the noise models and provides converters that allow switching between `cirq` circuits and those of `orquestra`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Core docs](https://zapatacomputing.github.io/orquestra-core/index.html).

For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Running on GPU

The parameters to configure GPU executions are supplied to `QSimSimulator` as `QSimOptions`. The details of these parameters can be found in [qsimcirq python interface](https://quantumai.google/qsim/cirq_interface#gpu_execution). Passing `use_gpu=True` will enable gpu. If you want to use [NVIDIA's cuStateVec](https://docs.nvidia.com/cuda/cuquantum/custatevec/index.html), please additionally pass `gpu_mode=1` as can be seen in the example below:

```
from orquestra.integrations.cirq.simulator import QSimSimulator

from qsimcirq import QSimOptions

qsim_options = QSimOptions(use_gpu=True, gpu_mode=1)

sim = QSimSimulator(qsim_options=qsim_options)
```

`CuStateVecSimulator` is using `QsimSimulator` and by default is set to `use_gpu=true` and `gpu_mode=1`. Below is an example of importing `CuStateVecSimulator`:

```
from orquestra.integrations.custatevec.simulator import CuStateVecSimulator

sim = CuStateVecSimulator()
```

## Development and contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).
