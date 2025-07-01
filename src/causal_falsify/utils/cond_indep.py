from typing import Optional
import numpy as np
from causallearn.utils.cit import CIT

"""
Multiple implementations of conditional independence tests to test null hypothesis:

    H0: X ⟂ Y | Z

"""


def _validate_inputs(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Validate input shapes for CI tests.
    """
    if not (x.ndim == 2 and x.shape[1] == 1):
        raise ValueError(
            f"Input x must be a 2D array with a single column (shape: (n_samples, 1)), but has shape {x.shape}."
        )
    if not (y.ndim == 2 and y.shape[1] == 1):
        raise ValueError(
            f"Input y must be a 2D array with a single column (shape: (n_samples, 1)), but has shape {y.shape}."
        )
    if not (z.ndim == 2):
        raise ValueError(
            f"Input z must be a 2D array with shape (n_samples, n_features), but has shape {z.shape}."
        )

    n_samples = x.shape[0]
    if not (y.shape[0] == n_samples == z.shape[0]):
        raise ValueError("All inputs must have the same number of rows.")


def kcit_rbf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[float]:
    """
    Kernel-based Conditional Independence Test (KCIT) with RBF kernels.

    Args:
        x: Array of shape (n_samples, 1).
        y: Array of shape (n_samples, 1).
        z: Array of shape (n_samples, n_covariates).

    Returns:
        p-value of the test if successful; None if an error occurred.
    """
    _validate_inputs(x, y, z)

    data = np.hstack([x, y, z])

    try:
        cit_obj = CIT(
            data,
            method="kci",
            kernelX="Gaussian",
            kernelY="Gaussian",
            kernelZ="Gaussian",
            approx=False,
            use_gp=True,
            est_width="median",
        )
        conditioning_set = list(range(2, data.shape[1]))
        pval = cit_obj(0, 1, conditioning_set)
        assert 0 <= pval <= 1, f"Invalid p-value computed: {pval}"
        return pval

    except AssertionError as e:
        print(f"Assertion error: {e}")
        return None


def fisherz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[float]:
    """
    Fisher's Z Conditional Independence Test.

    Args:
        x: Array of shape (n_samples, 1).
        y: Array of shape (n_samples, 1).
        z: Array of shape (n_samples, n_covariates).

    Returns:
        p-value of the test if successful; None if an error occurred.
    """
    _validate_inputs(x, y, z)

    data = np.hstack([x, y, z])

    try:
        cit_obj = CIT(data, method="fisherz")
        conditioning_set = list(range(2, data.shape[1]))
        pval = cit_obj(0, 1, conditioning_set)
        assert 0 <= pval <= 1, f"Invalid p-value computed: {pval}"
        return pval

    except AssertionError as e:
        print(f"Assertion error: {e}")
        return None
