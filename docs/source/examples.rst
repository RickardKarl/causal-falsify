Example Usage
-------------

Below is an example using the MINT algorithm. When calling ``MINT().test(...)``, it returns a p-value that can be interpreted as a statistical test of the joint null hypothesis: that unconfoundedness holds and the causal mechanisms are independent. A sufficiently small p-value indicates rejection of this null, suggesting that hidden confounding may be present. Note that HGIC and the transportability test have their corresponding classes ``HGIC`` and ``TransportabilityTest`` with the same ``test(...)`` function.

.. code-block:: python

   from causal_falsify.algorithms.mint import MINT
   from causal_falsify.utils.simulate_data import simulate_data

   # Generate a pd.DataFrame with synthetic data containing columns [X_0, X_1, S, A, Y]
   confounded_data = simulate_data(
       n_samples=250,
       conf_strength=1.0,
       n_envs=10,
       n_observed_confounders=2
   )

   # Initialize falsification algorithm and run it on the data
   mint_algorithm = MINT(binary_treatment=False)
   p_value = mint_algorithm.test(
       confounded_data,
       covariate_vars=["X_0", "X_1"],
       treatment_var="A",
       outcome_var="Y",
       source_var="S",
   )

   # A small p-value would imply that hidden confounding may be present
   print("p-value:", p_value)
   print("reject null of no hidden confounding at sign. level 5%:", p_value < 0.05)

