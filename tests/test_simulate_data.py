import numpy as np
import pandas as pd
import pytest

from causal_falsify.utils.simulate_data import (
    create_polynomial_representation,
    simulate_data,
)


# --------------------------
# Tests for create_polynomial_representation
# --------------------------


def test_create_polynomial_correct_shape():
    X = np.array([[1, 2], [3, 4]])
    degree = 3
    X_poly = create_polynomial_representation(X, degree)
    assert X_poly.shape == (2, 6), "Incorrect shape of polynomial features"


def test_create_polynomial_degree_1_equals_input():
    X = np.random.rand(10, 3)
    X_poly = create_polynomial_representation(X, degree=1)
    np.testing.assert_almost_equal(X, X_poly)


def test_create_polynomial_raises_on_non_array():
    with pytest.raises(TypeError):
        create_polynomial_representation([[1, 2], [3, 4]], degree=2)


def test_create_polynomial_raises_on_wrong_dim():
    with pytest.raises(ValueError):
        create_polynomial_representation(np.array([1, 2, 3]), degree=2)


def test_create_polynomial_raises_on_bad_degree():
    X = np.random.rand(5, 2)
    with pytest.raises(ValueError):
        create_polynomial_representation(X, 0)

    with pytest.raises(TypeError):
        create_polynomial_representation(X, "2")


def test_create_polynomial_raises_on_empty_input():
    with pytest.raises(ValueError):
        create_polynomial_representation(np.empty((0, 2)), 2)

    with pytest.raises(ValueError):
        create_polynomial_representation(np.empty((3, 0)), 2)


# --------------------------
# Tests for simulate_data
# --------------------------


def test_simulate_data_shape():
    df = simulate_data(n_samples=10, n_envs=3, n_observed_confounders=2, seed=42)
    expected_rows = 10 * 3
    expected_cols = 2 + 2 + 1  # A, Y, X_0, X_1, S
    assert df.shape == (expected_rows, expected_cols)
    assert set(["A", "Y", "X_0", "X_1", "S"]).issubset(df.columns)


def test_simulate_data_reproducibility():
    df1 = simulate_data(n_samples=5, seed=123)
    df2 = simulate_data(n_samples=5, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_simulate_data_env_column():
    df = simulate_data(n_samples=2, n_envs=4)
    assert df["S"].nunique() == 4
    assert set(df["S"].unique()) == {0, 1, 2, 3}


def test_simulate_data_raises_on_negative_samples():
    with pytest.raises(ValueError):
        simulate_data(n_samples=-5)
