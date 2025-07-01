import numpy as np
import pandas as pd
from typing import List, Dict
from jax import random, vmap, config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from causal_falsify.abstract import FalsificationAlgorithm
from causal_falsify.utils.mint import (
    create_polynomial_representation,
    bootstrap_model_fitting_jax,
    fit_outcome_model_jax,
    fit_treatment_model_jax,
    permutation_independence_test,
    bootstrapped_permutation_independence_test,
)


class MINT(FalsificationAlgorithm):
    def __init__(
        self,
        feature_representation: str = "linear",
        feature_representation_params: dict = {},
        binary_treatment: bool = False,
        min_samples_per_env: int = 25,
        independence_test_args: dict = {},
        n_bootstraps: int = 1000,
    ) -> None:
        """

        Mechanism INdependent Test (MINT) algorithm from
        "Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms"
        Karlsson and Krijthe, ICML 2025
        (https://arxiv.org/abs/2502.06231)

        Joint test for whether we have independence between causal mechanisms and unconfoundedness across sources.
        A rejection will falsify both conditions jointly.

        Args:
            feature_representation (str): Feature representation to use ("linear" or "poly").
            feature_representation_params (dict): Parameters for the feature representation.
            binary_treatment (bool): Whether the treatment is binary (currently not implemented).
            min_samples_per_env (int): Minimum number of samples required per environment.
            independence_test_args (dict): Arguments for the independence test.
            n_bootstraps (int): Number of bootstrap iterations. If None, no bootstrapping is used.
        """
        self.feature_representation = feature_representation
        self.feature_representation_params = feature_representation_params
        self.binary_treatment = binary_treatment
        self.min_samples_per_env = min_samples_per_env
        self.independence_test_args = independence_test_args
        self.n_bootstraps = n_bootstraps

        # Store last diagnostics for access via get_diagnostics()
        self._last_model_fit_diagnostics = None

    def test(
        self,
        data: pd.DataFrame,
        covariate_vars: List[str],
        treatment_var: str,
        outcome_var: str,
        source_var: str,
    ) -> float:
        """
        Perform falsification test for joint test of unconfoundedness and independence of causal mechanisms.

        Args:
            data (dict): Dictionary mapping environment name to pd.DataFrame.
            covariate_vars (list): List of covariate column names.
            treatment_var (str): Name of treatment column.
            outcome_var (str): Name of outcome column.
            source_var (str): Name of source/environment column.

        Returns:
            p-value
        """

        # Validate required columns
        required_cols = set(covariate_vars + [treatment_var, outcome_var, source_var])
        missing = required_cols.difference(data.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        # Extract arrays for the test
        outcome = data[[outcome_var]].values  # shape (n_samples, 1)
        treatment = data[[treatment_var]].values  # shape (n_samples, 1)
        source = data[[source_var]].values  # shape (n_samples, 1)
        covariates = data[covariate_vars].values  # shape (n_samples, n_covariates)

        n_environments = len(np.unique(source))
        coef_outcome_mech, coef_treatment_mech = [], []
        resampled_coef_outcome_mech, resampled_coef_treatment_mech = [], []

        model_fit_diagnostics = {
            "source_label": [],
            "outcome_model_mse": [],
            "treatment_model_mse": [],
        }

        # Prepare global feature mappings
        phi_outcome = self.get_feature_representation()
        phi_outcome_treatment = self.get_feature_representation()

        for source_label in np.unique(source):

            source_index = (source == source_label).squeeze()
            covariates_source = jnp.array(covariates[source_index, :])
            treatment_source = jnp.array(treatment[source_index, :])
            outcome_source = jnp.array(outcome[source_index, :])

            if covariates_source.shape[0] < self.min_samples_per_env:
                n_environments -= 1
                continue

            def add_intercept(term):
                return jnp.hstack([term, jnp.ones((term.shape[0], 1))])

            tf_outcome_source = add_intercept(phi_outcome(covariates_source))
            tf_outcome_treatment_source = add_intercept(
                phi_outcome_treatment(
                    jnp.concatenate([covariates_source, treatment_source], axis=1)
                )
            )

            params_outcome_mech, outcome_model_mse = fit_outcome_model_jax(
                tf_outcome_treatment_source, outcome_source
            )
            params_treatment_mech, treatment_model_mse = fit_treatment_model_jax(
                tf_outcome_source, treatment_source
            )

            coef_outcome_mech.append(params_outcome_mech)
            coef_treatment_mech.append(params_treatment_mech)

            model_fit_diagnostics["source_label"].append(source_label)
            model_fit_diagnostics["outcome_model_mse"].append(outcome_model_mse)
            model_fit_diagnostics["treatment_model_mse"].append(treatment_model_mse)

            if self.n_bootstraps:
                keys = random.split(random.PRNGKey(0), self.n_bootstraps)
                resampled_params = vmap(
                    bootstrap_model_fitting_jax,
                    in_axes=(None, None, None, None, 0),
                )(
                    outcome_source,
                    treatment_source,
                    tf_outcome_source,
                    tf_outcome_treatment_source,
                    keys,
                )
                resampled_coef_outcome_mech.append(resampled_params[0])
                resampled_coef_treatment_mech.append(resampled_params[1])

        coef_outcome_mech = np.array(jnp.vstack(coef_outcome_mech))
        coef_treatment_mech = np.array(jnp.vstack(coef_treatment_mech))

        if self.n_bootstraps:
            resampled_coef_outcome_mech = np.array(
                jnp.hstack(resampled_coef_outcome_mech)
            )
            resampled_coef_treatment_mech = np.array(
                jnp.hstack(resampled_coef_treatment_mech)
            )
            pval = self.run_bootstrapped_independence_test(
                coef_outcome_mech,
                coef_treatment_mech,
                resampled_coef_outcome_mech,
                resampled_coef_treatment_mech,
            )
        else:
            pval = self.run_independence_test(coef_outcome_mech, coef_treatment_mech)

        # save diagnostics from this run
        self._last_model_fit_diagnostics = model_fit_diagnostics

        return pval

    def get_diagnostics(self) -> Dict:
        """
        TODO Returns diagnostics from the last test run.

        Returns:
            dict: model_fit_diagnostics
        """
        return self._last_model_fit_diagnostics

    def get_feature_representation(self):
        if self.feature_representation == "linear":
            return lambda x: x
        elif self.feature_representation == "poly":
            return lambda x: create_polynomial_representation(
                x, **self.feature_representation_params
            )
        else:
            raise ValueError(
                f"Invalid feature representation: {self.feature_representation}"
            )

    def run_independence_test(self, data_x, data_y):
        return permutation_independence_test(data_x=data_x, data_y=data_y)

    def run_bootstrapped_independence_test(
        self, data_x, data_y, resampled_data_x, resampled_data_y
    ):
        return bootstrapped_permutation_independence_test(
            data_x=data_x,
            data_y=data_y,
            resampled_data_x=resampled_data_x,
            resampled_data_y=resampled_data_y,
        )
