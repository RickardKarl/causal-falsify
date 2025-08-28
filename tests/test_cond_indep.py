import numpy as np
import pytest
from causal_falsify.utils.cond_indep import (
    _validate_inputs,
    kcit_rbf,
    fisherz,
)


# --------------------------
# Parameterized tests for _validate_inputs
# --------------------------
@pytest.mark.parametrize(
    "x, y, z, error_msg",
    [
        (np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 3), None),
        (
            np.random.rand(10, 2),
            np.random.rand(10, 1),
            np.random.rand(10, 3),
            "Input x must be a 2D array with a single column",
        ),
        (
            np.random.rand(10, 1),
            np.random.rand(10, 2),
            np.random.rand(10, 3),
            "Input y must be a 2D array with a single column",
        ),
        (
            np.random.rand(10, 1),
            np.random.rand(10, 1),
            np.random.rand(10),
            "Input z must be a 2D array",
        ),
        (
            np.random.rand(10, 1),
            np.random.rand(9, 1),
            np.random.rand(10, 3),
            "All inputs must have the same number of rows",
        ),
    ],
)
def test_validate_inputs_param(x, y, z, error_msg):
    if error_msg is None:
        _validate_inputs(x, y, z)  # should not raise
    else:
        with pytest.raises(ValueError, match=error_msg):
            _validate_inputs(x, y, z)


# --------------------------
# Test conditional independence tests
# --------------------------

# List of seeds
seeds = [0, 1, 41, 123, 999]

# List of CI functions to test
ci_tests = [kcit_rbf, fisherz]


# -------------------------
# Combined test for independent case
# -------------------------
def test_ci_independence_hold():
    for ci_func in ci_tests:
        for seed in seeds:
            np.random.seed(seed)
            z = np.random.normal(0, 1, (100, 2))
            x = np.random.normal(0, 1, (100, 1))
            y = np.random.normal(0, 1, (100, 1))

            pval = ci_func(x, y, z)
            assert (
                0 <= pval <= 1
            ), f"{ci_func.__name__} failed with seed {seed}, pval={pval}"


# -------------------------
# Combined test for dependent case
# -------------------------
def test_ci_dependence_hold():
    for ci_func in ci_tests:
        for seed in seeds:
            np.random.seed(seed)
            z = np.random.normal(0, 1, (100, 2))
            x = np.random.normal(0, 1, (100, 1))
            y = 2 * x + np.random.normal(0, 1, (100, 1))  # make y dependent on x

            pval = ci_func(x, y, z)
            assert (
                0 <= pval <= 0.1
            ), f"{ci_func.__name__} failed with seed {seed}, pval={pval}"
