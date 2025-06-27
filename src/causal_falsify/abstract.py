import pandas as pd
from typing import List
from abc import ABC, abstractmethod


class FalsificationAlgorithm(ABC):
    def __init__(self) -> None:
        """
        Base class for all falsification algorithms.
        """
        super().__init__()

    @abstractmethod
    def test(
        self,
        data: pd.DataFrame,
        covariate_vars: List[str],
        treatment_var: str,
        outcome_var: str,
        source_var: str,
    ) -> float:
        """
        Run the falsification test to test null hypothesis of no unmeasured confounding in multi-source observational data.

        Args:
            data (pd.DataFrame): Input dataset.
            covariate_vars (List[str]): List of covariate column names.
            treatment_var (str): Treatment variable column name.
            outcome_var (str): Outcome variable column name.
            source_var (str): Source/environment indicator column_
        
        Returns:
            float: p-value from the test.
        """
        pass