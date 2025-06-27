# causal-falsify

[![PyPI](https://img.shields.io/pypi/v/causal-falsify.svg)](https://pypi.org/project/causal-falsify/)
[![License](https://img.shields.io/pypi/l/causal-falsify)](./LICENSE)

*causal-falsify: A Python library with algorithms for falsifying unconfoundedness assumption in a composite dataset from multiple sources.*

This library implements algorithms proposed in our two papers:

- **Detecting Hidden Confounding in Observational Data Using Multiple Environments** – NeurIPS 2023 ([pdf](https://arxiv.org/abs/2205.13935))  
- **Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms** – ICML 2025 ([pdf](https://arxiv.org/abs/2502.06231))

---

## 📦 Installation

Install from [PyPI](https://pypi.org/project/causal-falsify/):

```bash
pip install causal-falsify
```

---

## Example usage

An example with the Mechanism INdependence Test (MINT) algorithm.

```python
from causal_falsify.mint import MINT
from causal_falsify.utils.simulate_data import simulate_data

# Create a simulated pandas DataFrame containing where unmeasured confounding is present:
# - Observed pre-treatment covariates: ["X_0", "X_1"]
# - Source label: "S"
# - Treatment: "A"
# - Outcome: "Y"
confounded_data = simulate_data(
    n_samples=250, conf_strength=1.0, n_envs=10, n_observed_confounders=2
)

# Run the MINT algorithm
mint_algorithm = MINT()
p_value = mint_algorithm.test(
    confounded_data,
    covariate_vars=["X_0", "X_1"],
    treatment_var="A",
    outcome_var="Y",
    source_var="S",
)

# We are evaluating the joint null hypothesis of no unmeasured confounding 
# and independent causal mechanisms across sources.
# Reject the null if p-value < significance level (e.g., 0.05).
print("p-value:", p_value)
print("reject null:",  p_value < 0.05)

```

---

## 📄 Please cite our work if you use our package

The HGIC and MINT algorithms are based on two of our papers which you can cite as follows:

```bibtex
@article{karlsson2023detecting,
  title={Detecting hidden confounding in observational data using multiple environments},
  author={Karlsson, Rickard and Krijthe, Jesse},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={44280--44309},
  year={2023}
}

@inproceedings{karlsson2025falsification,
  title={Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms},
  author={Karlsson, Rickard KA and Krijthe, Jesse H},
  booktitle={International Conference on Machine Learning},
  organization={PMLR}
  year={2025},
}
```

---

## 📬 Contact

Created by [Rickard Karlsson](https://rickardkarlsson.com) – feel free to reach out!
