# causal-falsify

[![PyPI](https://img.shields.io/pypi/v/causal-falsify.svg)](https://pypi.org/project/causal-falsify/)
[![License](https://img.shields.io/pypi/l/causal-falsify)](./LICENSE)

*causal-falsify: A Python library with algorithms for falsifying unconfoundedness assumption in a composite dataset from multiple sources.*

This library implements algorithms proposed in our two papers based on testing independence of causal mechanisms:

- **Detecting Hidden Confounding in Observational Data Using Multiple Environments** – NeurIPS 2023 ([pdf](https://arxiv.org/abs/2205.13935))  
- **Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms** – ICML 2025 ([pdf](https://arxiv.org/abs/2502.06231))

---

## 📦 Installation

Install from [PyPI](https://pypi.org/project/causal-falsify/):

```bash
pip install causal-falsify
```

---

## Algorithms

We have implemented three falsification algorithms, which can be used complementarily:

- **Hierarchical Graphical Independence Constraint (HGIC) Test**:  
  This test jointly assesses whether transportability and unconfoundedness hold across sources. A rejection indicates that at least one of these conditions fails. The HGIC test is derived from specific d-separation using constraint-based causal discovery in a hierarchical causal graphical model.

- **Mechanism Independence Test (MINT)**:  
  Similar to the HGIC test, MINT jointly tests for transportability and unconfoundedness across sources. However, it makes a parametric linearity assumption, which greatly improves sample efficiency but may lead to false positives if the linear model is severely misspecified.

- **Transportability-Based Test**:  
  This alternative approach jointly tests for transportability and unconfoundedness across sources. A rejection here likewise indicates that at least one of these conditions does not hold.

### Example usage

An example with the MINT algorithm.

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
mint_algorithm = MINT(binary_treatment=False)
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
  author={Karlsson, Rickard and Krijthe, Jesse H},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={44280--44309},
  year={2023}
}

@inproceedings{karlsson2025falsification,
  title={Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms},
  author={Karlsson, Rickard and Krijthe, Jesse H},
  booktitle={International Conference on Machine Learning},
  organization={PMLR}
  year={2025},
}
```

---

## 🐛 Issues

If you encounter any bugs, unexpected behavior, or have questions about using the package, please don’t hesitate to [open an issue](https://github.com/RickardKarl/causal-falsify/issues).  

---

## 📬 Contact

Created by [Rickard Karlsson](https://rickardkarlsson.com) – feel free to reach out!
