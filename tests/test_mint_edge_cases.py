"""
Tests for MINT algorithm edge cases, including empty covariate sets.
Addresses issue #12: Implement edge case with empty covariate set for MINT
"""

import numpy as np
import pandas as pd
import pytest
from causal_falsify.algorithms.mint import MINT


def test_mint_with_empty_covariate_set():
    """
    Test that MINT handles empty covariate set correctly.

    When no covariates are provided, the algorithm should still work
    by fitting intercept-only models for the treatment mechanism
    and treatment+intercept for the outcome mechanism.
    """
    np.random.seed(42)

    # Create simple test data with multiple environments
    n_samples_per_env = 50
    n_envs = 5

    data_list = []
    for env_id in range(n_envs):
        # Simple data generation: T ~ N(0,1), Y ~ T + N(0,1)
        treatment = np.random.randn(n_samples_per_env)
        outcome = treatment + np.random.randn(n_samples_per_env) * 0.5

        env_data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'source': env_id
        })
        data_list.append(env_data)

    data = pd.concat(data_list, ignore_index=True)

    # Test with empty covariate list
    mint = MINT(
        binary_treatment=False,
        binary_outcome=False,
        min_samples_per_env=25,
        n_bootstraps=100  # Reduced for faster testing
    )

    # This should not raise an error
    p_value = mint.test(
        data=data,
        covariate_vars=[],  # Empty covariate set
        treatment_var='treatment',
        outcome_var='outcome',
        source_var='source'
    )

    # Validate p-value
    assert p_value is not None, "p-value should not be None"
    assert isinstance(p_value, (float, np.floating)), "p-value should be a float"
    assert 0 <= p_value <= 1, f"p-value should be in [0,1], got {p_value}"

    # Check diagnostics are available
    diagnostics = mint.get_diagnostics()
    assert diagnostics is not None
    assert len(diagnostics['source_label']) == n_envs, \
        f"Should have diagnostics for all {n_envs} environments"


def test_mint_with_empty_covariate_set_binary_treatment():
    """
    Test MINT with empty covariates and binary treatment.
    """
    np.random.seed(43)

    n_samples_per_env = 50
    n_envs = 5

    data_list = []
    for env_id in range(n_envs):
        # Binary treatment
        treatment = np.random.binomial(1, 0.5, n_samples_per_env)
        outcome = treatment * 2.0 + np.random.randn(n_samples_per_env) * 0.5

        env_data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'source': env_id
        })
        data_list.append(env_data)

    data = pd.concat(data_list, ignore_index=True)

    mint = MINT(
        binary_treatment=True,
        binary_outcome=False,
        min_samples_per_env=25,
        n_bootstraps=100
    )

    p_value = mint.test(
        data=data,
        covariate_vars=[],
        treatment_var='treatment',
        outcome_var='outcome',
        source_var='source'
    )

    assert 0 <= p_value <= 1


def test_mint_with_empty_covariate_set_binary_outcome():
    """
    Test MINT with empty covariates and binary outcome.
    """
    np.random.seed(44)

    n_samples_per_env = 50
    n_envs = 5

    data_list = []
    for env_id in range(n_envs):
        treatment = np.random.randn(n_samples_per_env)
        # Binary outcome based on treatment
        prob = 1 / (1 + np.exp(-treatment))
        outcome = np.random.binomial(1, prob)

        env_data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'source': env_id
        })
        data_list.append(env_data)

    data = pd.concat(data_list, ignore_index=True)

    mint = MINT(
        binary_treatment=False,
        binary_outcome=True,
        min_samples_per_env=25,
        n_bootstraps=100
    )

    p_value = mint.test(
        data=data,
        covariate_vars=[],
        treatment_var='treatment',
        outcome_var='outcome',
        source_var='source'
    )

    assert 0 <= p_value <= 1


def test_mint_insufficient_samples_error():
    """
    Test that MINT raises an informative error when no environments
    have enough samples.

    This addresses the issue where empty coefficient lists would cause
    a confusing "Need at least one array to concatenate" error.
    """
    np.random.seed(45)

    # Create data with very few samples per environment
    n_samples_per_env = 10
    n_envs = 3

    data_list = []
    for env_id in range(n_envs):
        treatment = np.random.randn(n_samples_per_env)
        outcome = treatment + np.random.randn(n_samples_per_env)
        covariate = np.random.randn(n_samples_per_env)

        env_data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'covariate': covariate,
            'source': env_id
        })
        data_list.append(env_data)

    data = pd.concat(data_list, ignore_index=True)

    # Set min_samples_per_env higher than available samples
    mint = MINT(
        binary_treatment=False,
        min_samples_per_env=50,  # More than the 10 samples per env
        n_bootstraps=100
    )

    # Should raise ValueError with informative message
    with pytest.raises(ValueError) as exc_info:
        mint.test(
            data=data,
            covariate_vars=['covariate'],
            treatment_var='treatment',
            outcome_var='outcome',
            source_var='source'
        )

    # Check that error message is informative
    error_msg = str(exc_info.value)
    assert "No environments have at least" in error_msg
    assert "50" in error_msg  # min_samples_per_env value
    assert "reduce min_samples_per_env" in error_msg or "more data" in error_msg


def test_mint_empty_covariates_with_polynomial_features():
    """
    Test that polynomial feature representation works with empty covariates.

    When covariates are empty, polynomial features should still work
    (degenerating to just the intercept).
    """
    np.random.seed(46)

    n_samples_per_env = 50
    n_envs = 5

    data_list = []
    for env_id in range(n_envs):
        treatment = np.random.randn(n_samples_per_env)
        outcome = treatment + np.random.randn(n_samples_per_env) * 0.5

        env_data = pd.DataFrame({
            'treatment': treatment,
            'outcome': outcome,
            'source': env_id
        })
        data_list.append(env_data)

    data = pd.concat(data_list, ignore_index=True)

    # Test with polynomial features
    mint = MINT(
        binary_treatment=False,
        feature_representation='poly',
        feature_representation_params={'degree': 2},
        min_samples_per_env=25,
        n_bootstraps=100
    )

    p_value = mint.test(
        data=data,
        covariate_vars=[],  # Empty covariates with poly features
        treatment_var='treatment',
        outcome_var='outcome',
        source_var='source'
    )

    assert 0 <= p_value <= 1
