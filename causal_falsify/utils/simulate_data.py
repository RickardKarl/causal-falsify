import numpy as np
import pandas as pd


def create_polynomial_representation(X, degree):
    """
    Generate polynomial features for the input array X up to a specified degree.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features).
    degree : int
        Degree of the polynomial features.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_features * degree) containing the polynomial features.
    """

    # Validate input type
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    if not isinstance(degree, int):
        raise TypeError("degree must be an integer.")

    if degree < 1:
        raise ValueError("degree must be a positive integer (>= 1).")

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        raise ValueError("X must have at least one sample and one feature.")

    # Create polynomial features
    poly_features = []
    for feature_idx in range(n_features):
        feature = X[:, feature_idx]
        poly_feature = np.column_stack([feature**d for d in range(1, degree + 1)])
        poly_features.append(poly_feature)

    X_poly = np.hstack(poly_features)
    return X_poly


def simulate_data(
    n_samples: int,
    degree: int = 1,
    conf_strength: float = 1.0,
    transportability_violation: float = 0.0,
    n_envs: int = 50,
    n_observed_confounders: int = 5,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulates synthetic data for causal inference experiments with multiple environments,
    observed confounders, and configurable treatment and outcome mechanisms.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate per environment.
    degree : int, optional
        Degree of polynomial transformation applied to observed confounders (default is 1, i.e., linear).
    conf_strength : float, optional
        Strength of confounding between treatment and outcome (default is 1.0).
    transportability_violation : float, optional
        Degree of violation in transportability across environments (default is 0.0).
    n_envs : int, optional
        Number of distinct environments to simulate (default is 50).
    n_observed_confounders : int, optional
        Number of observed confounders/features (default is 5).
    seed : int, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the simulated data with columns:
        - 'A': Treatment variable
        - 'Y': Outcome variable
        - 'X_0', ..., 'X_{n_observed_confounders-1}': Observed confounders
        - 'S': Environment index

    Notes
    -----
    - The function generates data for multiple environments, each with its own parameters.
    - Observed confounders can be transformed using polynomial features.
    - Unobserved confounding and transportability violations can be controlled via parameters.
    """

    rng = np.random.RandomState(seed)
    covar_list = [f"X_{i}" for i in range(n_observed_confounders)]

    x_transform = (
        (lambda x: x)
        if degree == 1
        else (lambda x: create_polynomial_representation(x, degree))
    )

    test_vector = rng.multivariate_normal(
        np.zeros(n_observed_confounders), np.eye(n_observed_confounders), size=(1)
    )
    feature_dim = x_transform(test_vector).shape[1]

    x_to_a_coef = rng.choice([-1.0, 1.0], size=(feature_dim, n_envs))
    x_to_y_coef = np.ones((feature_dim, n_envs), dtype=float)
    a_to_y_effect = np.ones((1, n_envs), dtype=float)

    intercept_a = rng.normal(0, 1.0, size=(1, n_envs))
    intercept_y = rng.normal(0, 1.0, size=(1, n_envs))

    mu_X = rng.normal(0, 1.0, size=(n_envs, n_observed_confounders))
    mu_U = rng.normal(0, 1.0, size=(n_envs, 1))
    X_cov = np.full((n_observed_confounders, n_observed_confounders), 0.1)
    np.fill_diagonal(X_cov, 2)
    sigma_U = 2.0

    all_data = []

    for i in range(n_envs):
        X = rng.multivariate_normal(
            mu_X[i], X_cov / np.sqrt(n_observed_confounders), size=n_samples
        )
        X_repr = x_transform(X)
        U = rng.normal(mu_U[i], sigma_U, size=(n_samples, 1))

        treatment_confounding = 1.0 if conf_strength != 0.0 else 0.0
        A = (
            intercept_a[:, i]
            + (X_repr @ x_to_a_coef[:, i]).reshape(-1, 1)
            + treatment_confounding * np.sum(np.abs(U), axis=1).reshape(-1, 1)
            + rng.normal(0, 0.5, size=(n_samples, 1))  # Treatment noise
        )

        Y = (
            transportability_violation * intercept_y[:, i]
            + (X_repr @ x_to_y_coef[:, i]).reshape(-1, 1)
            + conf_strength * np.sum(np.abs(U), axis=1).reshape(-1, 1)
            + (a_to_y_effect[:, i] * A).reshape(-1, 1)
            + rng.normal(0, 0.5, size=(n_samples, 1))  # Outcome noise
        )

        df_env = pd.DataFrame(
            np.concatenate([A, Y, X], axis=1),
            columns=["A", "Y"] + covar_list,
        )
        df_env["S"] = i
        all_data.append(df_env)

    df_all = pd.concat(all_data, ignore_index=True)
    return df_all
