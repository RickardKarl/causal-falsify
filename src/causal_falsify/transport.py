from causal_falsify.abstract import FalsificationAlgorithm
import numpy as np
import pandas as pd
from typing import List

from causal_falsify.utils.cond_indep import (
    kcit_rbf,
    fisherz,
)


class TransportabilityTest(FalsificationAlgorithm):
    def __init__(
        self,
        cond_indep_test: str = "kcit_rbf",
        max_sample_size: int = np.inf,
    ) -> None:
        """

        Transportability-based test akin to the benchmarking framework in
        "Using Trial and Observational Data to Assess Effectiveness:
        Trial Emulation, Transportability, Benchmarking, and Joint Analysis"
        Dahabreh et al., 2024

        Joint test for whether we have transportability and unconfoundedness across sources.
        A rejection will falsify both conditions jointly.

        Args:
            cond_indep_test (str): CI test to use: "kcit_rbf" or "fisherz".
            max_sample_size (int): Max samples for testing to control runtime.
        """
        super().__init__()
        if max_sample_size <= 0:
            raise ValueError("max_sample_size must be larger than zero")

        self.cond_indep_test = cond_indep_test
        self.max_sample_size_test = max_sample_size

    def test(
        self,
        data: pd.DataFrame,
        covariate_vars: List[str],
        treatment_var: str,
        outcome_var: str,
        source_var: str,
    ) -> float:
        """
        Perform falsification test for joint test of unconfoundedness and transportability.

        Args:
            data (pd.DataFrame): DataFrame containing all required columns.
            covariate_vars (List[str]): Covariate column names to condition on.
            treatment_var (str): Treatment column name.
            outcome_var (str): Outcome column name.
            source_var (str): Source/environment indicator column name.

        Returns:
            float: p-value of the conditional independence test.
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

        # Subsample if necessary
        if outcome.shape[0] > self.max_sample_size_test:
            outcome, source, covariates, treatment = self.subsample_data(
                outcome, source, covariates, treatment
            )

        # Select conditional independence test function
        if self.cond_indep_test == "kcit_rbf":
            test_func = kcit_rbf
        elif self.cond_indep_test == "fisherz":
            test_func = fisherz
        else:
            raise ValueError(f"Unsupported cond_indep_test: {self.cond_indep_test}")

        # Test if outcome is independent of source conditional on covariates and treatment
        conditioning_vars = np.hstack([covariates, treatment])
        pval = test_func(outcome, source, conditioning_vars)

        return pval

    def subsample_data(self, outcome, source, covariates, treatment):
        """
        Subsample data to limit size, preserving source distribution proportions.
        """
        unique_sources, counts = np.unique(source, return_counts=True)
        proportions = counts / counts.sum()

        sampled_indices = []
        for src_value, proportion in zip(unique_sources, proportions):
            src_indices = np.where(source.flatten() == src_value)[0]
            n_samples = min(
                len(src_indices), int(np.round(proportion * self.max_sample_size_test))
            )
            sampled_indices.extend(
                np.random.choice(src_indices, n_samples, replace=False)
            )

        if len(sampled_indices) > self.max_sample_size_test:
            sampled_indices = np.random.choice(
                sampled_indices, self.max_sample_size_test, replace=False
            )

        return (
            outcome[sampled_indices, :],
            source[sampled_indices, :],
            covariates[sampled_indices, :],
            treatment[sampled_indices, :],
        )
