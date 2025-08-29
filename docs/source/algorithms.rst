Algorithms
----------

We have (so far) implemented three falsification algorithms.

- **Hierarchical Graphical Independence Constraint (HGIC) Test**  
  Jointly assesses unconfoundedness and independence of causal mechanisms across environments.  

- **Mechanism Independence Test (MINT)**  
  Parametric version with a linearity assumption for better sample efficiency.  
  Implementation based on independence of causal mechanisms across environments.

- **Transportability-Based Test**  
  Tests joint transportability and unconfoundedness across environments. **Important:** Transportability holds when there is no edge between environment and outcome in the causal graph, meaning that that this test is not advised if one suspect that edge to be present in your data.

These algorithms are explained more in-depth in two papers, we encourage you to read these papers to learn more:

1. *Detecting Hidden Confounding in Observational Data Using Multiple Environments* – Karlsson and Krijthe,  
   `NeurIPS 2023 (PDF) <https://arxiv.org/abs/2205.13935>`_

2. *Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms* – Karlsson and Krijthe,   
   `ICML 2025 (PDF) <https://arxiv.org/abs/2502.06231>`_

