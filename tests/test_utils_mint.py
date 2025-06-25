import numpy as np
import pytest
import jax.numpy as jnp
from jax import random

from causal_falsify.utils.mint import (
    create_polynomial_representation,
    compute_offdiag_block_frobnorm,
    permutation_independence_test,
    bootstrapped_permutation_independence_test,
    validate_matrix,
    fit_logistic_regression,
    fit_linear_regression,
    cross_val_mse,
    fit_outcome_model_jax,
    fit_treatment_model_jax,
    bootstrap_model_fitting_jax,
    resample_until_enough_unique,
)


def test_create_polynomial_representation_basic():
    X = np.array([[1, 2], [3, 4]])
    # degree must be > 1, else error
    with pytest.raises(ValueError):
        create_polynomial_representation(X, degree=1)

    # basic polynomial expansion without sklearn
    poly = create_polynomial_representation(X, degree=2, use_sklearn=False)
    assert poly.shape == (2, 4)  # 2 features * degree 2
    # with sklearn interaction_only
    poly2 = create_polynomial_representation(
        X, degree=2, use_sklearn=True, interaction_only=True
    )
    assert poly2.shape[0] == X.shape[0]


def test_validate_matrix_good_and_bad():
    good = np.array([[1.0, 2.0], [3.0, 4.0]])
    validate_matrix(good)  # should not raise

    with pytest.raises(AssertionError):
        validate_matrix(np.array([1, 2, 3]))  # not 2D

    with pytest.raises(AssertionError):
        validate_matrix(np.array([[np.nan, 1], [2, 3]]))

    with pytest.raises(AssertionError):
        validate_matrix(np.array([[np.inf, 1], [2, 3]]))


def test_compute_offdiag_block_frobnorm_shape_and_value():
    np.random.seed(0)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    norm = compute_offdiag_block_frobnorm(x, y)
    assert isinstance(norm, float)
    assert norm >= 0


def test_permutation_independence_test_basic():
    np.random.seed(0)
    x = np.random.randn(10, 2)
    y = np.random.randn(10, 2)
    p_val = permutation_independence_test(
        x, y, n_bootstraps=10, random_state=np.random.RandomState(0)
    )
    assert 0 <= p_val <= 1


def test_bootstrapped_permutation_independence_test_basic():
    np.random.seed(0)
    x = np.random.randn(10, 2)
    y = np.random.randn(10, 2)
    resampled_x = np.random.randn(3, 10, 2)
    resampled_y = np.random.randn(3, 10, 2)
    p_val = bootstrapped_permutation_independence_test(
        x, y, resampled_x, resampled_y, random_state=np.random.RandomState(0)
    )
    assert 0 <= p_val <= 1


def test_fit_logistic_regression_basic():
    np.random.seed(0)
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    T = jnp.array([1, -1, 1, -1])
    params = fit_logistic_regression(X, T, alpha=0.01)
    assert params.shape == (X.shape[1],)


def test_fit_linear_regression_basic():
    np.random.seed(0)
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = jnp.array([1, 2, 3, 4])
    params = fit_linear_regression(X, Y)
    assert params.shape == (X.shape[1],)


def test_cross_val_mse_basic():
    np.random.seed(0)
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Y = jnp.array([1, 2, 3, 4, 5])
    mse = cross_val_mse(X, Y, fit_linear_regression, num_folds=5)
    assert mse >= 0


def test_fit_outcome_and_treatment_model_jax():
    np.random.seed(0)
    X = jnp.array(np.random.randn(10, 3))
    T = jnp.array(np.random.randn(10))
    Y = jnp.array(np.random.randn(10))

    params_outcome, mse_outcome = fit_outcome_model_jax(X, Y)
    params_treatment, mse_treatment = fit_treatment_model_jax(X, T)

    assert params_outcome.shape == (X.shape[1],)
    assert mse_outcome >= 0
    assert params_treatment.shape == (X.shape[1],)
    assert mse_treatment >= 0


def test_resample_until_enough_unique():
    key = random.PRNGKey(0)
    indices = resample_until_enough_unique(key, n_resamples=10, min_sample_size=5)
    assert len(indices) == 10
    assert len(set(indices.tolist())) >= 5


def test_bootstrap_model_fitting_jax_basic():
    key = random.PRNGKey(0)
    np.random.seed(0)
    Y = jnp.array(np.random.randn(15))
    T = jnp.array(np.random.randn(15))
    tf_X = jnp.array(np.random.randn(15, 3))
    tf_XT = jnp.array(np.random.randn(15, 3))
    params_outcome, params_treatment = bootstrap_model_fitting_jax(
        Y, T, tf_X, tf_XT, key
    )
    assert params_outcome.shape == (tf_XT.shape[1],)
    assert params_treatment.shape == (tf_X.shape[1],)
