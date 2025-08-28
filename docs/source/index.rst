causal-falsify
===============

.. image:: https://img.shields.io/pypi/v/causal-falsify.svg
   :target: https://pypi.org/project/causal-falsify
.. image:: https://img.shields.io/pypi/l/causal-falsify
   :target: ./LICENSE



A Python library with algorithms for falsifying the unconfoundedness assumption
in multi-environment (or, multi-source) datasets. The source code is available on
GitHub: https://github.com/RickardKarl/causal-falsify

Below is an example causal directed acyclic graph where these algorithms could be used to detect the presence of the hidden confounder. Different environments could, for instance, correspond to distinct time periods or locations in which the data was collected.

.. image:: _static/causal_dag.jpeg
   :alt: Causal DAG
   :width: 50%
   :align: center
   
Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install causal-falsify

Algorithms
----------

We have implemented three falsification algorithms:

- **Hierarchical Graphical Independence Constraint (HGIC) Test**  
  Jointly assesses unconfoundedness and independence of causal mechanisms across environments.  

- **Mechanism Independence Test (MINT)**  
  Parametric version with a linearity assumption for better sample efficiency.  
  Implementation based on independence of causal mechanisms across environments.

- **Transportability-Based Test**  
  Tests joint transportability and unconfoundedness across environments. Important: Transportability assumes that there is no edge between environment and outcome in the causal graph.

These algorithms are implementations presented in two papers, we encourage you to read these papers to learn more in-depth about the algorithms:

1. *Detecting Hidden Confounding in Observational Data Using Multiple Environments* – Karlsson and Krijthe,  
   `NeurIPS 2023 (PDF) <https://arxiv.org/abs/2205.13935>`_

2. *Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms* – Karlsson and Krijthe,   
   `ICML 2025 (PDF) <https://arxiv.org/abs/2502.06231>`_

API Reference & Example Usage
-----------------------------

.. toctree::
   :maxdepth: 3

   causal_falsify
 


Below is an example using the MINT algorithm. When calling MINT.test(...), it returns a p-value that can be interpreted as a statistical test of the joint null hypothesis: that unconfoundedness holds and the causal mechanisms are independent. A sufficiently small p-value indicates rejection of this null, suggesting that hidden confounding may be present. Note that HGIC and the transportability test have similar test(...) functions.

.. code-block:: python

   from causal_falsify.algorithms.mint import MINT
   from causal_falsify.utils.simulate_data import simulate_data

   confounded_data = simulate_data(
       n_samples=250,
       conf_strength=1.0,
       n_envs=10,
       n_observed_confounders=2
   )

   mint_algorithm = MINT(binary_treatment=False)
   p_value = mint_algorithm.test(
       confounded_data,
       covariate_vars=["X_0", "X_1"],
       treatment_var="A",
       outcome_var="Y",
       source_var="S",
   )

   print("p-value:", p_value)
   print("reject null of no hidden confounding:", p_value < 0.05)

Citing
------

Please cite our work if you use this package:

.. code-block:: bibtex

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
     organization={PMLR},
     year={2025},
   }

Issues & Contact
----------------

If you encounter any bugs, unexpected behavior, or have questions, feel free to
`open an issue <https://github.com/RickardKarl/causal-falsify/issues>`_.


