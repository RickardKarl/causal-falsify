import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import grad, jit, random, lax
from sklearn.preprocessing import PolynomialFeatures


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
    X: jnp.ndarray, T: jnp.ndarray, alpha: float = 1e-3
) -> jnp.ndarray:
    """
    Fit a logistic regression model using JAX and gradient descent with ridge regularization.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Transformed feature matrix, including intercept term if desired.
    T : array-like, shape (n_samples,)
        Target variable. Must be binary (typically 0/1 or -1/1).
    alpha : float, optional (default=1e-3)
        Regularization strength for ridge (L2) penalty. Set to 0 for no regularization.

    Returns
    -------
    params : jax.numpy.ndarray, shape (n_features,)
        Fitted logistic regression coefficients.
    """

    # Define logistic regression loss with optional regularization
    def logistic_loss(params, X, T, alpha):
        logits = X @ params
        loss = jnp.mean(jnp.log(1 + jnp.exp(-T * logits)))  # Logistic loss
        if alpha > 0:
            loss += alpha * jnp.sum(params**2)  # Regularization term (Ridge)
        return loss

    # Initial guess for parameters (weights)
    init_params = jnp.zeros(X.shape[1])

    # Compute the gradient of the loss function
    loss_grad = grad(logistic_loss)

    # Minimize the loss using gradient descent
    def update(params, X, T, alpha, learning_rate=0.1):
        grads = loss_grad(params, X, T, alpha)
        return params - learning_rate * grads

    # Fit the model by iterating updates
    params = init_params
    for _ in range(1000):  # Set max iterations for gradient descent
        params = update(params, X, T, alpha)

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


def fit_outcome_model_jax(
    tf_XT: jnp.ndarray, Y: jnp.ndarray
) -> tuple[jnp.ndarray, float]:
    """
    Fits a linear regression outcome model using JAX and evaluates its performance via cross-validation.

    Parameters
    ----------
    tf_XT : jax.numpy.ndarray
        The design matrix of shape (n_samples, n_features), where each row represents a sample and each column a feature.
    Y : jax.numpy.ndarray
        The outcome vector of shape (n_samples,), containing the target values.

    Returns
    -------
    params_outcome : jax.numpy.ndarray
        The fitted model parameters of shape (n_features,).
    model_mse : float
        The mean squared error (MSE) of the model estimated via 5-fold cross-validation.

    Raises
    ------
    AssertionError
        If the number of samples is not greater than the number of features.
    """

    assert tf_XT.shape[0] > tf_XT.shape[1], "need more samples than features"

    # Fit the outcome model using model_fn
    params_outcome = fit_linear_regression(tf_XT, Y)

    # Perform cross-validation for model diagnostic using the same model_fn
    model_mse = cross_val_mse(tf_XT, Y, fit_linear_regression, num_folds=5)

    return params_outcome.T, model_mse


def fit_treatment_model_jax(
    tf_X: jnp.ndarray, T: jnp.ndarray
) -> tuple[jnp.ndarray, float]:
    """
    Fits a linear regression treatment model using JAX and evaluates its performance via cross-validation.

    Parameters
    ----------
    tf_X : jnp.ndarray
        The feature matrix of shape (n_samples, n_features).
    T : jnp.ndarray
        The treatment assignment vector of shape (n_samples,).

    Returns
    -------
    params_treatment : jnp.ndarray
        The fitted model parameters, transposed.
    model_mse : float
        The mean squared error from cross-validation.

    Raises
    ------
    AssertionError
        If the number of samples is not greater than the number of features.

    Notes
    -----
    Uses `fit_linear_regression` for model fitting and `cross_val_mse` for cross-validation.
    """

    assert tf_X.shape[0] > tf_X.shape[1], "need more samples than features"

    # Fit the model and get parameters
    params_treatment = fit_linear_regression(tf_X, T)

    # Perform cross-validation for model diagnostic using the same model_fn
    model_mse = cross_val_mse(tf_X, T, fit_linear_regression, num_folds=5)

    return params_treatment.T, model_mse


@jit
def bootstrap_model_fitting_jax(
    Y: jnp.ndarray, T: jnp.ndarray, tf_X: jnp.ndarray, tf_XT: jnp.ndarray, key
):
    """
    Fits outcome and treatment models on a bootstrap resample of the data using JAX.

    This function performs bootstrap resampling of the input data arrays using JAX's random
    number generation for reproducibility. It ensures that the resampled data contains enough
    unique samples for model estimation. The function then fits outcome and treatment models
    on the resampled data and returns the fitted parameters.

    Parameters
    ----------
    Y : jnp.ndarray
        Outcome variable array of shape (n_samples,).
    T : jnp.ndarray
        Treatment variable array of shape (n_samples,).
    tf_X : jnp.ndarray
        Transformed covariate array for the treatment model of shape (n_samples, n_features).
    tf_XT : jnp.ndarray
        Transformed covariate array for the outcome model of shape (n_samples, n_features_outcome).
    key : jax.random.PRNGKey
        JAX random key for reproducibility.

    Returns
    -------
    resampled_params_outcome : Any
        Fitted parameters of the outcome model on the resampled data.
    resampled_params_treatment : Any
        Fitted parameters of the treatment model on the resampled data.

    Raises
    ------
    AssertionError
        If the number of samples is not sufficient for model estimation.
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
    resampled_params_outcome, _ = fit_outcome_model_jax(resampled_tf_XT, resampled_Y)
    resampled_params_treatment, _ = fit_treatment_model_jax(resampled_tf_X, resampled_T)

    return resampled_params_outcome, resampled_params_treatment


def resample_until_enough_unique(subkey, n_resamples, min_sample_size):
    """
    Repeatedly resamples indices with replacement until at least `min_sample_size` unique indices are obtained.

    Parameters
    ----------
    subkey : jax.random.PRNGKey
        The random key used for generating random numbers.
    n_resamples : int
        The number of indices to sample in each resampling iteration.
    min_sample_size : int
        The minimum number of unique indices required in the resampled set.

    Returns
    -------
    resampled_indices : jax.numpy.ndarray
        An array of shape `(n_resamples,)` containing the resampled indices, guaranteed to have at least
        `min_sample_size` unique values.

    Notes
    -----
    This function uses a while loop to repeatedly resample indices until the number of unique indices
    in the sample meets or exceeds `min_sample_size`. The sampling is performed with replacement.
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
        return count_unique(resampled_indices) < min_sample_size

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
