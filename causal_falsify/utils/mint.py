import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.scipy.linalg import solve
from jax import grad, jit, random, lax
from sklearn.preprocessing import PolynomialFeatures
import warnings


def create_polynomial_representation(
    X, degree, use_sklearn=False, interaction_only=False
) -> np.ndarray:
    """
    Generate a polynomial feature representation of the input data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to be transformed into polynomial features.
    degree : int
        The degree of the polynomial features to be generated. Must be greater than 1.
    use_sklearn : bool, optional, default=False
        If True, use sklearn's PolynomialFeatures for transformation. If False, generate polynomial features manually (only powers of individual features, no cross-terms).
    interaction_only : bool, optional, default=False
        If True and `use_sklearn` is True, only interaction features are produced: features that are products of at most `degree` distinct input features (no powers of single features). Has no effect if `use_sklearn` is False.

    Returns
    -------
    X_poly : ndarray of shape (n_samples, n_output_features)
        The matrix of polynomial features.

    Raises
    ------
    ValueError
        If `degree` is less than or equal to 1.

    Notes
    -----
    - When `use_sklearn` is False, only powers of individual features are generated (no interaction/cross terms).
    - When `use_sklearn` is True, both interaction and power terms are generated according to the parameters.
    """
    if degree <= 1:
        raise ValueError("Degree must be larger than 1.")

    if interaction_only and not use_sklearn:
        print("Warning: interaction_only has no effect as use_sklearn = False.")

    if use_sklearn:
        return PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=False
        ).fit_transform(X)

    else:
        n_features = X.shape[1]

        # Create an empty list to store polynomial features
        poly_features = []

        # Iterate over each feature
        for feature_idx in range(n_features):
            # Create polynomial features for the current feature
            feature = X[:, feature_idx]
            poly_feature = np.column_stack([feature**d for d in range(1, degree + 1)])
            poly_features.append(poly_feature)

        # Stack the polynomial features horizontally
        X_poly = np.hstack(poly_features)
        return X_poly


###############################################################
# Test based on computing Frobenius norm of off-diagonal block
###############################################################


def compute_offdiag_block_frobnorm(data_x, data_y) -> float:
    """
    Compute the Frobenius norm of the off-diagonal block of the covariance matrix between two datasets.

    Given two datasets with the same number of samples, this function concatenates them,
    computes the covariance matrix, extracts the off-diagonal block corresponding to the
    covariances between the two datasets, and returns its Frobenius norm.

    Parameters
    ----------
    data_x : np.ndarray
        A 2D array of shape (n_samples, n_features_x) representing the first dataset.
    data_y : np.ndarray
        A 2D array of shape (n_samples, n_features_y) representing the second dataset.

    Returns
    -------
    float
        The Frobenius norm of the off-diagonal block of the covariance matrix between `data_x` and `data_y`.

    Raises
    ------
    AssertionError
        If the number of samples (first dimension) in `data_x` and `data_y` do not match.
    ValueError
        If the input matrices are not valid as determined by `validate_matrix`.

    Notes
    -----
    The off-diagonal block refers to the submatrix of the covariance matrix that captures
    the covariances between the features of `data_x` and `data_y`.
    """

    dim_x, dim_y = data_x.shape[1], data_y.shape[1]
    assert data_x.shape[0] == data_y.shape[0], "first dimension be the same"
    coefs = np.hstack([data_x, data_y])

    validate_matrix(coefs)

    covariance_matrix = np.cov(coefs, rowvar=False)
    offdiag_block = covariance_matrix[:dim_x, dim_x:]
    assert offdiag_block.shape == (dim_x, dim_y)

    return np.linalg.norm(offdiag_block, "fro")


def permutation_independence_test(
    data_x: np.ndarray, data_y: np.ndarray, n_bootstraps: int = 1000, random_state=None
) -> float:
    """
    Performs a permutation-based independence test between two datasets.

    This function tests the null hypothesis that `data_x` and `data_y` are independent
    by comparing the observed off-diagonal block Frobenius norm to the distribution
    obtained by permuting `data_x`. The p-value is estimated as the proportion of
    permuted statistics greater than the observed statistic.

    Parameters
    ----------
    data_x : np.ndarray
        The first dataset, with samples along the first axis.
    data_y : np.ndarray
        The second dataset, with samples along the first axis.
    n_bootstraps : int, optional
        Number of permutations to perform (default is 1000).
    random_state : np.random.RandomState or None, optional
        Random state for reproducibility. If None, a new RandomState is created.

    Returns
    -------
    float
        The estimated p-value for the independence test.

    Notes
    -----
    Requires the function `compute_offdiag_block_frobnorm` to compute the test statistic.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    observed_frob_norm = compute_offdiag_block_frobnorm(data_x, data_y)

    resampled_frob_norm = np.zeros((n_bootstraps, 1))
    for j in range(n_bootstraps):

        # permute rows in coef_t
        permuted_data_x = random_state.permutation(data_x)  # permutates on first axis
        resampled_frob_norm[j] = compute_offdiag_block_frobnorm(permuted_data_x, data_y)

    return np.mean(observed_frob_norm < resampled_frob_norm)


def bootstrapped_permutation_independence_test(
    data_x: np.ndarray,
    data_y: np.ndarray,
    resampled_data_x: np.ndarray,
    resampled_data_y: np.ndarray,
    random_state=None,
) -> float:
    """
    Performs a bootstrapped permutation independence test between two datasets.

    This function computes the observed off-diagonal block Frobenius norm between
    `data_x` and `data_y`, then compares it to the distribution of norms obtained
    by permuting the resampled versions of `data_x` and `data_y`. The returned value
    is the proportion of times the observed statistic is less than the bootstrapped
    statistics, which can be interpreted as a p-value for the independence test.

    Parameters
    ----------
    data_x : np.ndarray
        The original data array for variable X, of shape (n_samples, n_features_x).
    data_y : np.ndarray
        The original data array for variable Y, of shape (n_samples, n_features_y).
    resampled_data_x : np.ndarray
        Bootstrapped samples of `data_x`, of shape (n_bootstraps, n_samples, n_features_x).
    resampled_data_y : np.ndarray
        Bootstrapped samples of `data_y`, of shape (n_bootstraps, n_samples, n_features_y).
    random_state : np.random.RandomState or None, optional
        Random state for reproducibility. If None, a new RandomState is created.

    Returns
    -------
    float
        The proportion of bootstrapped statistics greater than the observed statistic,
        representing the p-value for the independence test.

    Raises
    ------
    AssertionError
        If the number of bootstraps in `resampled_data_x` and `resampled_data_y` do not match.

    Notes
    -----
    This function relies on `compute_offdiag_block_frobnorm` to compute the test statistic.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    n_bootstraps = resampled_data_x.shape[0]

    assert resampled_data_x.shape[:1] == resampled_data_y.shape[:1]

    observed_frob_norm = compute_offdiag_block_frobnorm(data_x, data_y)

    resampled_frob_norm = np.zeros((n_bootstraps, 1))
    for j in range(n_bootstraps):

        permuted_resampled_data_x = random_state.permutation(
            resampled_data_x[j, :, :].squeeze()
        )

        resampled_frob_norm[j] = compute_offdiag_block_frobnorm(
            permuted_resampled_data_x, resampled_data_y[j, :, :].squeeze()
        )

    return np.mean(observed_frob_norm < resampled_frob_norm)


##########################################
# Utils
##########################################


def validate_matrix(matrix: np.ndarray):
    """
    Validates that the input matrix is a proper 2-dimensional NumPy array without NaN or infinite values.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to validate.

    Raises
    ------
    AssertionError
        If the input is not a NumPy array.
        If the matrix contains NaN values.
        If the matrix contains infinite values.
        If the matrix is not 2-dimensional.
    """
    # Assert that the input is a NumPy array
    assert isinstance(matrix, np.ndarray), "Input must be a NumPy array."

    # Assert no NaN values
    assert not jnp.isnan(matrix).any(), f"Matrix contains NaN values: {matrix}"

    # Assert no infinite values
    assert not np.isinf(matrix).any(), "Matrix contains infinite values."

    # Assert proper dimensionality
    assert matrix.ndim == 2, "Matrix must be 2-dimensional."


###############################################################
# Methods for estimating linear models
###############################################################


def fit_logistic_regression(
    X: jnp.ndarray, Y: jnp.ndarray, alpha: float = 0
) -> jnp.ndarray:
    """
    Fit a logistic regression model using JAX and gradient descent (binary cross-entropy).

    Implementation notes
    --------------------
    - Uses a numerically-stable logits-based binary cross-entropy loss.
    - Inputs `X` and `Y` are converted to JAX arrays and cast to a float dtype
      inside the function (`float32` by default).
    - Training is executed inside a JIT-compiled loop using `jax.lax.fori_loop` for
      efficiency. The function itself is decorated with `@jit`.
    - L2 regularization is applied as `alpha * sum(params**2)`. Passing `alpha=0`
      results in no regularization.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Design matrix (can include an intercept column).
    Y : array-like, shape (n_samples,)
        Binary target values in {0, 1} (the implementation expects 0/1 labels).
    alpha : float, optional (default=0)
        Regularization strength for ridge (L2) penalty. May be a traced JAX scalar
        when called from JITted code; the implementation avoids Python boolean checks
        on `alpha`.

    Returns
    -------
    params : jax.numpy.ndarray, shape (n_features,)
        Fitted logistic regression coefficients (dtype `float32`).

    Notes
    -----
    - The number of gradient-descent iterations and learning rate are currently
      hard-coded (1000 iterations, learning rate 0.1). If you need tunable training
      behaviour, consider adding explicit arguments or switching to an optimizer
      library such as `optax`.
    """

    # Ensure inputs are JAX arrays with float dtype
    X = jnp.asarray(X)
    Y = jnp.asarray(Y).astype(jnp.float32)

    # Define binary cross-entropy (BCE) loss from logits (numerically stable)
    # Uses formulation: BCE(logits, y) = max(l,0) - l*y + log(1 + exp(-abs(l)))
    def logistic_loss(parms, x, y, alp):
        logits = x @ parms
        # stable per-example loss
        per_example = (
            jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )
        loss = jnp.mean(per_example)
        # Always add regularization term (alpha can be 0)
        loss = loss + alp * jnp.sum(parms**2)
        return loss

    # Initial guess for parameters (weights)
    init_params = jnp.zeros(X.shape[1], dtype=jnp.float32)

    # Compute the gradient of the loss function
    # Use a traced loop via lax.fori_loop for efficient JIT compilation
    def body(i, parms):
        grads = grad(logistic_loss)(parms, X, Y, alpha)
        return parms - 0.1 * grads

    num_iters = 1000
    params = lax.fori_loop(0, num_iters, body, init_params)

    return params


def fit_linear_regression(
    X: jnp.ndarray, Y: jnp.ndarray, alpha: float = 0.0
) -> jnp.ndarray:
    """
    Fit a linear regression model using JAX.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Transformed feature matrix, including intercept term if desired.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target variable.
    alpha : float, optional (default=0)
        Regularization strength for ridge (L2) penalty. Set to 0 for ordinary least squares.

    Returns
    -------
    params : jax.numpy.ndarray of shape (n_features,) or (n_features, n_targets)
        Fitted linear regression coefficients.
    """

    I = jnp.eye(X.shape[1])  # Identity matrix for regularization
    I = I.at[-1, -1].set(0)  # Exclude intercept from regularization
    params = solve(X.T @ X + alpha * I, X.T @ Y)
    return params


def cross_val_mse(X: jnp.ndarray, Y: jnp.ndarray, model_fn, num_folds: int) -> float:
    """
    Perform k-fold cross-validation and compute the mean squared error (MSE).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Transformed feature matrix, including intercept term if desired.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target variable.
    model_fn : callable
        Function to fit the model. Should take (X_train, Y_train) and return fitted parameters.
    num_folds : int
        Number of folds for cross-validation.

    Returns
    -------
    float
        Mean squared error averaged across all folds.
    """
    n = X.shape[0]
    fold_size = n // num_folds
    mse_list = []

    for i in range(num_folds):
        # Split data into training and validation sets
        val_indices = jnp.arange(i * fold_size, (i + 1) * fold_size)
        train_indices = jnp.concatenate(
            [jnp.arange(0, i * fold_size), jnp.arange((i + 1) * fold_size, n)]
        )

        X_train, X_val = X[train_indices], X[val_indices]
        Y_train, Y_val = Y[train_indices], Y[val_indices]

        # Fit the model and get parameters using the training set
        params = model_fn(X_train, Y_train)

        # Compute MSE on validation set
        preds = X_val @ params
        mse = jnp.mean((Y_val - preds) ** 2)
        mse_list.append(mse)

    return jnp.mean(jnp.array(mse_list))


def fit_model_jax(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    binary_response: bool = False,
) -> tuple[jnp.ndarray, float]:
    """
    Fit a nuisance model (linear or logistic) and evaluate its performance via cross-validation.

    This function selects between `fit_linear_regression` and `fit_logistic_regression`
    based on the `binary_response` flag and returns the fitted parameters along with
    a cross-validated mean-squared error diagnostic.

    Important
    ---------
    - `binary_response` is treated as a plain Python boolean here. When calling
      `fit_model_jax` from JIT-compiled code, prefer passing function objects
      directly to the caller (see `bootstrap_model_fitting_jax`) to avoid tracing
      boolean values.

    Parameters
    ----------
    X : jax.numpy.ndarray
        Design matrix of shape `(n_samples, n_features)`.
    Y : jax.numpy.ndarray
        Target vector of shape `(n_samples,)`.
    binary_response : bool, optional
        If True, fit a logistic regression model; otherwise fit a linear model.

    Returns
    -------
    params_outcome : jax.numpy.ndarray
        The fitted model parameters (shape `(n_features,)`).
    model_mse : float
        Cross-validated mean squared error for diagnostic purposes.
    """

    assert X.shape[0] > X.shape[1], "need more samples than features"

    # Fit the outcome model using model_fn
    if binary_response:
        model_fn = fit_logistic_regression
    else:
        model_fn = fit_linear_regression

    params_outcome = model_fn(X=X, Y=Y)

    # Perform cross-validation for model diagnostic using the same model_fn
    try:
        model_mse = cross_val_mse(X, Y, model_fn, num_folds=3)
    except Exception as e:
        warnings.warn(f"Cross-validation failed: {str(e)}")
        model_mse = np.nan

    return params_outcome.T, model_mse


@partial(jit, static_argnames=["outcome_model_fn", "treatment_model_fn"])
def bootstrap_model_fitting_jax(
    Y: jnp.ndarray,
    T: jnp.ndarray,
    tf_X: jnp.ndarray,
    tf_XT: jnp.ndarray,
    outcome_model_fn,
    treatment_model_fn,
    key,
):
    """
    Fit outcome and treatment models on a bootstrap resample of the data.

    This function performs bootstrap resampling (with replacement) using JAX random
    primitives and then fits the provided model functions on the resampled data.

    Parameters
    ----------
    Y : jnp.ndarray
        Outcome array of shape `(n_samples,)`.
    T : jnp.ndarray
        Treatment array of shape `(n_samples,)`.
    tf_X : jnp.ndarray
        Transformed covariate matrix for the treatment model of shape `(n_samples, n_features)`.
    tf_XT : jnp.ndarray
        Transformed covariate matrix for the outcome model of shape `(n_samples, n_features_outcome)`.
    outcome_model_fn : callable
        Callable that fits an outcome model. Signature should be `fn(X, Y)` and return
        `(params, mse)` where `params` is an array of fitted coefficients.
    treatment_model_fn : callable
        Callable that fits a treatment model. Same interface as `outcome_model_fn`.
    key : jax.random.PRNGKey
        JAX PRNGKey used for resampling.

    Returns
    -------
    resampled_params_outcome, resampled_params_treatment : tuple
        The fitted parameters for outcome and treatment models on the resampled data.

        Notes
        -----
        - The function expects `outcome_model_fn` and `treatment_model_fn` to be plain
            Python callables (they can be `functools.partial` wrappers). The function
            is JIT-compiled here, and the two callable arguments are treated as static
            via `static_argnums` so they must be passed as Python callables (not JAX
            tracers / arrays).
        - This function uses JAX operations for resampling and is JIT-compiled with
            the model callables static to avoid tracing Python callables.
    """

    # Resample indices using JAX's random module for reproducibility
    key, subkey = random.split(key)  # Split the key to get a new one for resampling

    min_sample_size_needed_for_estimation = tf_X.shape[1] + 1
    assert (
        tf_X.shape[0] > min_sample_size_needed_for_estimation
    ), f"need more samples than {min_sample_size_needed_for_estimation}"
    resampled_indices = resample_until_enough_unique(
        subkey, Y.shape[0], min_sample_size_needed_for_estimation
    )

    # Resample the data
    resampled_Y = Y[resampled_indices]
    resampled_T = T[resampled_indices]
    resampled_tf_X = tf_X[resampled_indices]
    resampled_tf_XT = tf_XT[resampled_indices]

    # Fit outcome and treatment models on resampled data
    resampled_params_outcome, _ = outcome_model_fn(resampled_tf_XT, resampled_Y)
    resampled_params_treatment, _ = treatment_model_fn(resampled_tf_X, resampled_T)

    return resampled_params_outcome, resampled_params_treatment


def resample_until_enough_unique(subkey, n_resamples, min_sample_size):
    """
    Repeatedly resample indices (with replacement) until the sample contains at least
    `min_sample_size` unique indices.

    Parameters
    ----------
    subkey : jax.random.PRNGKey
        PRNG key for JAX random operations.
    n_resamples : int
        Number of indices to sample in each iteration (sample size).
    min_sample_size : int
        Minimum required number of unique indices in the resampled set.

    Returns
    -------
    resampled_indices : jax.numpy.ndarray
        Integer array of shape `(n_resamples,)` containing resampled indices. The
        returned array is guaranteed to contain at least `min_sample_size` distinct values
        when the function returns.

    Notes
    -----
    - The function uses `jax.lax.while_loop` internally to remain compatible with JIT
      tracing. The condition and body are expressed with JAX primitives.
    - If `min_sample_size > n_resamples` the loop cannot succeed; the caller should
      ensure `min_sample_size <= n_resamples` to avoid an infinite loop.
    """
    # Initial resampling
    resampled_indices = random.choice(
        subkey, n_resamples, shape=(n_resamples,), replace=True
    )

    def count_unique(x):
        x = jnp.sort(x)
        return 1 + (x[1:] != x[:-1]).sum()

    # Define condition function for while loop
    def condition_fn(state):
        _, resampled_indices = state
        # Check if unique indices are below the threshold
        # Use jnp.asarray() to ensure the result is a JAX array that can be used in lax.while_loop
        return jnp.asarray(count_unique(resampled_indices) < min_sample_size)

    # Define body function for while loop
    def body_fn(state):
        subkey, _ = state
        # Resample and update state
        subkey, new_subkey = random.split(subkey)
        resampled_indices = random.choice(
            new_subkey, n_resamples, shape=(n_resamples,), replace=True
        )
        return (subkey, resampled_indices)

    # Initial state: (key, resampled_indices)
    state = (subkey, resampled_indices)

    # Apply while loop until the condition is met
    _, resampled_indices = lax.while_loop(condition_fn, body_fn, state)

    return resampled_indices
