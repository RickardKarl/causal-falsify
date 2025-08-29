from causal_falsify.algorithms.abstract import AbstractFalsificationAlgorithm
import numpy as np
import pandas as pd
from typing import List

from causal_falsify.utils.cond_indep import (
    kcit_rbf,
    fisherz,
)


class TransportabilityTest(AbstractFalsificationAlgorithm):
    def __init__(
        self,
        cond_indep_test: str = "kcit_rbf",
        max_sample_size: int = np.inf,
        seed: int | None = None,
    ) -> None:
        """
        Transportability-based test.

        Inspired by the benchmarking framework in:

            Dahabreh et al., 2024. "Using Trial and Observational Data to Assess Effectiveness:
            Trial Emulation, Transportability, Benchmarking, and Joint Analysis"

        Performs a joint test for transportability and unconfoundedness across sources.
        A rejection indicates that both conditions are likely violated.

        Parameters
        ----------
        cond_indep_test : str
            Conditional independence test to use. Options are:
            - 'kcit_rbf': Kernel-based conditional independence test with RBF kernel.
            - 'fisherz': Fisher z-transform test for linear conditional independence.
        max_sample_size : int, optional
            Maximum number of samples to use during testing. Helps control runtime for
            large datasets. Defaults to None (use all samples).
        seed : int, optional
            Used when subsampling data (necessary if max_sample_size is smaller than total dataset size)

        Raises
        ------
        ValueError
            If `cond_indep_test` is not one of the supported options.
        """

        super().__init__()
        if max_sample_size <= 0:
            raise ValueError("max_sample_size must be larger than zero")

        self.cond_indep_test = cond_indep_test
        self.max_sample_size_test = max_sample_size
        self.rng = np.random.default_rng(seed)

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
            float: p-value of the test; low p-value implies unmeasured confounding may be present.
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

    def subsample_data(
        self,
        outcome: np.ndarray,
        source: np.ndarray,
        covariates: np.ndarray,
        treatment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Subsample data to limit the number of samples while preserving the source distribution.

        Parameters
        ----------
        outcome : np.ndarray of shape (n_samples, 1)
            Outcome variable for each sample.
        source : np.ndarray of shape (n_samples, 1)
            Source indicator for each sample.
        covariates : np.ndarray of shape (n_samples, n_covariates)
            Observed covariates for each sample.
        treatment : np.ndarray of shape (n_samples, 1)
            Treatment assignment for each sample.

        Returns
        -------
        outcome_sub : np.ndarray of shape (n_subsamples, 1)
            Subsampled outcomes.
        source_sub : np.ndarray of shape (n_subsamples, 1)
            Subsampled source indicators.
        covariates_sub : np.ndarray of shape (n_subsamples, n_covariates)
            Subsampled covariates.
        treatment_sub : np.ndarray of shape (n_subsamples, 1)
            Subsampled treatment assignments.

        Notes
        -----
        - The method ensures that each source is represented approximately
        proportionally to its frequency in the original data.
        - If the total number of selected samples exceeds `self.max_sample_size_test`,
        a random subset of the selected samples is drawn to enforce the limit.
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
                self.rng.choice(src_indices, n_samples, replace=False)
            )

        if len(sampled_indices) > self.max_sample_size_test:
            sampled_indices = self.rng.choice(
                sampled_indices, self.max_sample_size_test, replace=False
            )

        return (
            outcome[sampled_indices, :],
            source[sampled_indices, :],
            covariates[sampled_indices, :],
            treatment[sampled_indices, :],
        )
