################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import json
import os

import cirq
import pytest

from orquestra.integrations.cirq.noise.basic import (
    get_amplitude_damping,
    get_asymmetric_depolarize,
    get_depolarizing_channel,
    get_phase_damping,
    load_noise_model_from_json,
)


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


# Testing Depolarizing noise model
@pytest.mark.parametrize(
    "T, t_gate",
    [
        [10e-6, 10e-9],
        [10e-7, 10e-9],
        [56e-6, 10e-7],
    ],
)
def test_get_depolarizing_channel(T, t_gate):
    noise_model = get_depolarizing_channel(T, t_gate)
    assert isinstance(noise_model, cirq.ops.DepolarizingChannel)


@pytest.mark.parametrize(
    "T, t_gate",
    [
        [0.0001, -0.1],
        [-10e-7, 10e-9],
        [56e-6, -10e-7],
        [0, 10e-9],
        [10e-5, 0],
        [-56e-6, -10e-7],
    ],
)
def test_get_depolarizing_channel_fails_with_unphysical_values(T, t_gate):
    with pytest.raises(AssertionError):
        _ = get_depolarizing_channel(T, t_gate)


# Testing PTA noise model
@pytest.mark.parametrize(
    "T_1, T_2, t_gate",
    [
        [10e-6, 5e-6, 10e-9],
        [10e-7, 5e-7, 10e-9],
        [56e-6, 125e-6, 10e-7],
    ],
)
def test_get_get_asymmetric_depolarize(T_1, T_2, t_gate):
    noise_model = get_asymmetric_depolarize(T_1, T_2, t_gate)
    assert isinstance(noise_model, cirq.ops.AsymmetricDepolarizingChannel)


@pytest.mark.parametrize(
    "T_1, T_2, t_gate",
    [
        [-10e-7, 10e-4, 10e-9],
        [56e-6, -10e-7, 10e-9],
        [0, 10e-8, 10e-9],
        [10e-5, 0, 10e-9],
        [-56e-6, -10e-7, 0],
    ],
)
def test_get_asymmetric_depolarize_fails_with_unphysical_values(T_1, T_2, t_gate):
    with pytest.raises(AssertionError):
        _ = get_asymmetric_depolarize(T_1, T_2, t_gate)


# Testing amplitude damping model
@pytest.mark.parametrize(
    "T_1, t_gate",
    [
        [10e-6, 10e-9],
        [10e-7, 10e-9],
        [56e-6, 10e-7],
    ],
)
def test_get_amplitude_damping(T_1, t_gate):
    noise_model = get_amplitude_damping(T_1, t_gate)
    assert isinstance(noise_model, cirq.ops.AmplitudeDampingChannel)


@pytest.mark.parametrize(
    "T_1, t_gate",
    [
        [0.0001, -0.1],
        [-10e-7, 10e-9],
        [56e-6, -10e-7],
        [0, 10e-9],
        [10e-5, 0],
        [-56e-6, -10e-7],
    ],
)
def test_get_amplitude_damping_fails_with_unphysical_values(T_1, t_gate):
    with pytest.raises(AssertionError):
        _ = get_amplitude_damping(T_1, t_gate)


# Testing dephasing model
@pytest.mark.parametrize(
    "T_2, t_gate",
    [
        [10e-6, 10e-9],
        [10e-7, 10e-9],
        [56e-6, 10e-7],
    ],
)
def test_get_phase_damping(T_2, t_gate):
    noise_model = get_phase_damping(T_2, t_gate)
    assert isinstance(noise_model, cirq.ops.PhaseDampingChannel)


@pytest.mark.parametrize(
    "T_2, t_gate",
    [
        [0.0001, -0.1],
        [-10e-7, 10e-9],
        [56e-6, -10e-7],
        [0, 10e-9],
        [10e-5, 0],
        [-56e-6, -10e-7],
    ],
)
def test_get_phase_damping_fails_with_unphysical_values(T_2, t_gate):
    with pytest.raises(AssertionError):
        _ = get_phase_damping(T_2, t_gate)


@pytest.mark.parametrize(
    "serialized_model",
    [
        {"cirq_type": "DepolarizingChannel", "p": 0.5},
        '{"cirq_type": "DepolarizingChannel","p": 0.5}',
    ],
)
def test_load_noise_model_from_json(serialized_model):
    noise_model = load_noise_model_from_json(serialized_model)
    assert isinstance(noise_model, cirq.ops.DepolarizingChannel)
