import warnings
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from sklearn.exceptions import ConvergenceWarning

from causal_falsify.abstract import FalsificationAlgorithm
from causal_falsify.utils.cond_indep import (
    kcit_rbf,
    fisherz,
)

warnings.simplefilter("ignore", ConvergenceWarning)


class HGIC(FalsificationAlgorithm):
    def __init__(
        self,
        cond_indep_test: str = "kcit_rbf",
        max_tests: int = -1,
        min_test_sample_size: int = 25,
        method_pval_combination: str = "tippett",
    ) -> None:
        """

        Hierarchical Graphical Independence Constraint (HGIC) algorithm from
        "Detecting Hidden Confounding in Observational Data Using Multiple Environments"
        Karlsson and Krijthe, NeurIPS 2023
        (https://arxiv.org/abs/2205.13935)

        Joint test for whether we have independence between causal mechanisms and unconfoundedness across sources.
        A rejection will falsify both conditions jointly.

        Args:
            cond_indep_test (str): CI test to use: "kcit_rbf" or "fisherz".
            max_tests (int): Maximum number of pairwise tests to perform.
            min_test_sample_size (int): Minimum sources per test.
            method_pval_combination (str): Method to combine p-values ("tippett", "fisher" or "stouffer").
        """
        super().__init__()

        if max_tests == 0:
            raise ValueError("max_tests must be non-zero (use -1 for unlimited).")

        self.cond_indep_test = cond_indep_test
        self.max_tests = max_tests
        self.min_test_sample_size = min_test_sample_size
        self.method_pval_combination = method_pval_combination
        # self.feature_scaling = self.independence_test_args.pop("feature_scaling", False)

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

        # Partition data by source variable
        grouped_data = {
            str(source_label): df for source_label, df in data.groupby(source_var)
        }

        # Sort sources by sample size
        sample_sizes = sorted(
            [(source_label, len(df)) for source_label, df in grouped_data.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        if len(sample_sizes) < 2:
            raise ValueError("At least two sources are required for the test.")

        max_samples_available = sample_sizes[1][1]
        max_tests = self.max_tests if self.max_tests > 0 else max_samples_available

        pval_list, samples_used, df_list = [], [], []
        n = 0

        while n < max_samples_available - 1:
            valid_sources = [source for source, size in sample_sizes if size > n + 1]
            if len(valid_sources) < self.min_test_sample_size:
                n += 2
                continue

            df_tmp = self._construct_pair_df(
                grouped_data,
                covariate_vars,
                treatment_var,
                outcome_var,
                valid_sources,
                n,
            )
            df_list.append(df_tmp)
            n += 2

            if len(df_list) >= max_tests:
                break

        if not df_list:
            raise ValueError("No valid sample pairs available for testing.")

        cond_vars = (
            [f"{v}_i" for v in covariate_vars]
            + [f"{v}_j" for v in covariate_vars]
            + ["treatment_i"]
        )

        if self.cond_indep_test == "kcit_rbf":
            test_func = kcit_rbf
        elif self.cond_indep_test == "fisherz":
            test_func = fisherz
        else:
            raise ValueError(f"Unsupported cond_indep_test: {self.cond_indep_test}")

        for df in df_list:
            pval = test_func(
                df["treatment_j"].values.reshape(-1, 1),
                df["outcome_i"].values.reshape(-1, 1),
                df[cond_vars].values,
            )

            if np.isnan(pval):
                print("Warning: Computed NaN p-value.")
            else:
                pval_list.append(pval)
                samples_used.append(len(df))

        if not pval_list:
            raise RuntimeError("No valid p-values were computed.")

        weights = (
            [np.sqrt(n) for n in samples_used]
            if self.method_pval_combination == "stouffer"
            else None
        )
        _, combined_pval = combine_pvalues(
            pval_list, method=self.method_pval_combination, weights=weights
        )

        return combined_pval

    def _construct_pair_df(
        self,
        data: Dict[str, pd.DataFrame],
        covariates: List[str],
        treatment_var: str,
        outcome_var: str,
        sources: List[str],
        index: int,
    ) -> pd.DataFrame:
        """
        Construct a pairwise sample DataFrame for statistical testing.

        Args:
            data (Dict[str, pd.DataFrame]): Environment data.
            sources (List[str]): List of source keys to sample from.
            covariates (List[str]): List of observed covariates.
            index (int): Starting index for sampling.

        Returns:
            pd.DataFrame: Constructed sample DataFrame.
        """
        get = lambda source, col: data[source][col].values

        df_data = {
            "treatment_i": np.array(
                [get(source, treatment_var)[index] for source in sources]
            ),
            "outcome_i": np.array(
                [get(source, outcome_var)[index] for source in sources]
            ),
            "treatment_j": np.array(
                [get(source, treatment_var)[index + 1] for source in sources]
            ),
            "outcome_j": np.array(
                [get(source, outcome_var)[index + 1] for source in sources]
            ),
        }

        for var in covariates:
            df_data[f"{var}_i"] = np.array(
                [get(source, var)[index] for source in sources]
            )
            df_data[f"{var}_j"] = np.array(
                [get(source, var)[index + 1] for source in sources]
            )

        return pd.DataFrame(data=df_data)
