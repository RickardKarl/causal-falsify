Algorithms
==========

We have implemented three falsification algorithms so far, whose documentation is found :doc:`here<api/causal_falsify.algorithms>`.

Hierarchical Graphical Independence Constraint (HGIC) Test
-----------------------------------------------------------

This algorithm jointly assesses unconfoundedness and the independence of causal mechanisms across environments, as introduced in  
*Detecting Hidden Confounding in Observational Data Using Multiple Environments* by Karlsson and Krijthe,  
`NeurIPS 2023 (PDF) <https://arxiv.org/abs/2205.13935>`_. 

The test can indicate whether either the causal mechanisms are not independently changing across environments, or if unconfoundedness across all environments does not hold. The method is implemented by verifying a specific d-separation through conditional independence testing.  

.. The implementation can be found in the :doc:`HGIC implementation <api/causal_falsify.algorithms.hgic>`.

Mechanism Independence Test (MINT)
----------------------------------

Similar to HGIC, MINT also jointly assesses unconfoundedness and independence of causal mechanisms across environments.

However, it follows a two-stage procedure: first fitting parametric nuisance models for the causal mechanisms, and then testing independence between them across environments. While this introduces a functional assumption, it often provides better sample efficiency.  
The full method is described in *Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms* – Karlsson and Krijthe,  
`ICML 2025 (PDF) <https://arxiv.org/abs/2502.06231>`_.  

.. The implementation can be found in the :doc:`MINT implementation <api/causal_falsify.algorithms.mint>`.

Transportability-Based Test
---------------------------

This algorithm tests joint transportability and unconfoundedness across environments, which is also discussed in-depth in the references above. It can indicate whether either transportability between environments fails, or if unconfoundedness across all environments does not hold.

.. The implementation can be found in the :doc:`TransportabilityTest implementation <api/causal_falsify.algorithms.transport>`.

**Important:** Transportability assumes there is no edge between the environment and the outcome in the causal graph. Therefore, this test is **not recommended** if such an edge is suspected in your data.

