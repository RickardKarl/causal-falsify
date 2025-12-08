import numpy as np
import pytest
from collections import namedtuple
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="The optimal value found for dimension.*is close to the specified upper bound.*",
)

from causal_falsify.algorithms.mint import MINT
from causal_falsify.algorithms.hgic import HGIC
from causal_falsify.algorithms.transport import TransportabilityTest
from causal_falsify.utils.simulate_data import simulate_data

# Seeds for reproducibility
seeds = range(0, 10)

# Default dataset parameters
DEFAULT_N_SAMPLES = 25
DEFAULT_N_ENVS = 25
DEFAULT_N_CONFOUNDERS = 2
DEFAULT_CONFOUNDING_STRENGTH = 1.0
DEFAULT_NONLINEAR_DEGREE = 2


MethodTestCase = namedtuple(
    "MethodTestCase",
    [
        "method_class",
        "method_kwargs",
        "conf_strength",
        "degree",
        "expect_significant",
        "n_envs",
    ],
)


def make_test_cases():
    cases = []

    # Default n_envs for most methods
    default_envs = DEFAULT_N_ENVS
    hgic_envs = 10 * DEFAULT_N_ENVS

    # MINT linear
    cases.extend(
        [
            MethodTestCase(
                MINT,
                {"binary_treatment": False, "feature_representation": "linear"},
                DEFAULT_CONFOUNDING_STRENGTH,
                1,
                True,
                default_envs,
            ),
            MethodTestCase(
                MINT,
                {"binary_treatment": False, "feature_representation": "linear"},
                0.0,
                1,
                False,
                default_envs,
            ),
        ]
    )

    # MINT poly
    cases.extend(
        [
            MethodTestCase(
                MINT,
                {
                    "binary_treatment": False,
                    "feature_representation": "poly",
                    "feature_representation_params": {
                        "degree": DEFAULT_NONLINEAR_DEGREE
                    },
                },
                DEFAULT_CONFOUNDING_STRENGTH,
                DEFAULT_NONLINEAR_DEGREE,
                True,
                default_envs,
            ),
            MethodTestCase(
                MINT,
                {
                    "binary_treatment": False,
                    "feature_representation": "poly",
                    "feature_representation_params": {
                        "degree": DEFAULT_NONLINEAR_DEGREE
                    },
                },
                0.0,
                DEFAULT_NONLINEAR_DEGREE,
                False,
                default_envs,
            ),
        ]
    )

    # HGIC
    cases.extend(
        [
            MethodTestCase(
                HGIC,
                {"cond_indep_test": "fisherz", "max_tests": 1},
                DEFAULT_CONFOUNDING_STRENGTH,
                1,
                True,
                hgic_envs,
            ),
            MethodTestCase(
                HGIC,
                {"cond_indep_test": "fisherz", "max_tests": 1},
                0.0,
                1,
                False,
                hgic_envs,
            ),
            MethodTestCase(
                HGIC,
                {"cond_indep_test": "kcit_rbf", "max_tests": 1},
                DEFAULT_CONFOUNDING_STRENGTH,
                DEFAULT_NONLINEAR_DEGREE,
                True,
                hgic_envs,
            ),
            MethodTestCase(
                HGIC,
                {"cond_indep_test": "kcit_rbf", "max_tests": 1},
                0.0,
                DEFAULT_NONLINEAR_DEGREE,
                False,
                hgic_envs,
            ),
        ]
    )

    # Transportability
    cases.extend(
        [
            MethodTestCase(
                TransportabilityTest,
                {"cond_indep_test": "fisherz"},
                DEFAULT_CONFOUNDING_STRENGTH,
                1,
                True,
                default_envs,
            ),
            MethodTestCase(
                TransportabilityTest,
                {"cond_indep_test": "fisherz"},
                0.0,
                1,
                False,
                default_envs,
            ),
            MethodTestCase(
                TransportabilityTest,
                {"cond_indep_test": "kcit_rbf", "max_sample_size": 250, "seed": 42},
                DEFAULT_CONFOUNDING_STRENGTH,
                DEFAULT_NONLINEAR_DEGREE,
                True,
                default_envs,
            ),
            MethodTestCase(
                TransportabilityTest,
                {"cond_indep_test": "kcit_rbf", "max_sample_size": 250, "seed": 42},
                0.0,
                DEFAULT_NONLINEAR_DEGREE,
                False,
                default_envs,
            ),
        ]
    )

    return cases


@pytest.mark.parametrize("test_case", make_test_cases())
def test_methods_run_with_no_error(test_case):

    np.random.seed(seeds[0])
    data = simulate_data(
        n_samples=DEFAULT_N_SAMPLES,
        degree=test_case.degree,
        conf_strength=test_case.conf_strength,
        n_envs=test_case.n_envs,
        n_observed_confounders=DEFAULT_N_CONFOUNDERS,
    )

    algorithm = test_case.method_class(**test_case.method_kwargs)
    pval = algorithm.test(
        data,
        covariate_vars=["X_0", "X_1"],
        treatment_var="A",
        outcome_var="Y",
        source_var="S",
    )

    # Validate p-value
    assert (
        pval is not None
    ), f"{test_case.method_class.__name__} p-value is None, args={test_case.method_kwargs}"
    assert isinstance(
        pval, (float, np.floating)
    ), f"{test_case.method_class.__name__} p-value type invalid, args={test_case.method_kwargs}"
    assert (
        0 <= pval <= 1
    ), f"{test_case.method_class.__name__} p-value out of range, args={test_case.method_kwargs}"


@pytest.mark.parametrize("test_case", make_test_cases())
@pytest.mark.localonly
def test_methods_valid_output(test_case):
    p_values = []

    for seed in seeds:
        np.random.seed(seed)
        data = simulate_data(
            n_samples=DEFAULT_N_SAMPLES,
            degree=test_case.degree,
            conf_strength=test_case.conf_strength,
            n_envs=test_case.n_envs,
            n_observed_confounders=DEFAULT_N_CONFOUNDERS,
        )

        algorithm = test_case.method_class(**test_case.method_kwargs)
        pval = algorithm.test(
            data,
            covariate_vars=["X_0", "X_1"],
            treatment_var="A",
            outcome_var="Y",
            source_var="S",
        )

        # Validate p-value
        assert (
            pval is not None
        ), f"{test_case.method_class.__name__} p-value is None, args={test_case.method_kwargs}"
        assert isinstance(
            pval, (float, np.floating)
        ), f"{test_case.method_class.__name__} p-value type invalid, args={test_case.method_kwargs}"
        assert (
            0 <= pval <= 1
        ), f"{test_case.method_class.__name__} p-value out of range, args={test_case.method_kwargs}"

        p_values.append(pval)

    mean_pval = np.mean(p_values)
    if test_case.expect_significant:
        assert mean_pval < 0.25, (
            f"Mean p-value for {test_case.method_class.__name__} {test_case.method_kwargs}: mean_pval={mean_pval} | "
            f"conf_strength={test_case.conf_strength}, degree={test_case.degree}, "
            f"n_envs={test_case.n_envs}, n_samples={DEFAULT_N_SAMPLES}, n_observed_confounders={DEFAULT_N_CONFOUNDERS}, "
            f"expect_significant={test_case.expect_significant}"
        )
    else:
        assert mean_pval > 0.25, (
            f"Mean p-value for {test_case.method_class.__name__} {test_case.method_kwargs}: mean_pval={mean_pval} | "
            f"conf_strength={test_case.conf_strength}, degree={test_case.degree}, "
            f"n_envs={test_case.n_envs}, n_samples={DEFAULT_N_SAMPLES}, n_observed_confounders={DEFAULT_N_CONFOUNDERS}, "
            f"expect_significant={test_case.expect_significant}"
        )
