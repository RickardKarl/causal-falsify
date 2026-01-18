# Fix for Issue #12: Implement edge case with empty covariate set for MINT

## Summary

This fix addresses GitHub issue #12 and also resolves the related bug described in issue #7, where the MINT algorithm would crash with a confusing error message when encountering edge cases.

## Problems Fixed

### 1. Empty Covariate Set Handling
**Issue**: When `covariate_vars=[]`, the MINT algorithm would fail because:
- Polynomial feature generation couldn't handle empty arrays
- Independence test functions expected 2D arrays but received 1D arrays for scalar parameters
- Model fitting returned 1D arrays that needed to be 2D for downstream processing

**Solution**:
- Added proper handling for empty covariates in `create_polynomial_representation()`
- Updated `compute_offdiag_block_frobnorm()` to auto-reshape 1D arrays to 2D
- Modified `fit_model_jax()` to always return 2D parameter arrays

### 2. Insufficient Samples Error (Related to Issue #7)
**Issue**: When all environments had fewer samples than `min_samples_per_env`, the coefficient lists would be empty, causing:
```
ValueError: Need at least one array to concatenate
```

**Solution**: Added validation after the environment loop to provide a clear, actionable error message:
```python
if len(coef_outcome_mech) == 0:
    raise ValueError(
        f"No environments have at least {self.min_samples_per_env} samples. "
        f"Found {len(np.unique(source))} environment(s) but all were skipped. "
        f"Either reduce min_samples_per_env or provide more data per environment."
    )
```

## Files Modified

### 1. `causal_falsify/algorithms/mint.py`
- **Lines 155-164**: Refactored feature transformation to explicitly handle empty covariates
- **Lines 198-204**: Added validation for empty coefficient lists with informative error message

### 2. `causal_falsify/utils/mint.py`
- **Lines 56-58**: Added empty covariate handling in `create_polynomial_representation()`
- **Lines 115-119**: Added automatic 1D to 2D reshaping in `compute_offdiag_block_frobnorm()`
- **Lines 486-492**: Ensured `fit_model_jax()` always returns 2D parameter arrays

### 3. `tests/test_mint_edge_cases.py` (New File)
Added comprehensive test suite covering:
- Empty covariate set with continuous treatment/outcome
- Empty covariate set with binary treatment
- Empty covariate set with binary outcome
- Empty covariate set with polynomial features
- Insufficient samples error handling

## Test Results

All new tests pass:
```
tests/test_mint_edge_cases.py::test_mint_with_empty_covariate_set PASSED
tests/test_mint_edge_cases.py::test_mint_with_empty_covariate_set_binary_treatment PASSED
tests/test_mint_edge_cases.py::test_mint_with_empty_covariate_set_binary_outcome PASSED
tests/test_mint_edge_cases.py::test_mint_insufficient_samples_error PASSED
tests/test_mint_edge_cases.py::test_mint_empty_covariates_with_polynomial_features PASSED
```

All existing tests continue to pass:
```
tests/test_methods_on_data.py::test_methods_run_with_no_error - 12 passed
```

## Use Case

This fix enables users to test for unmeasured confounding using MINT even when no observed covariates are available. This is a valid scenario where:
- **Treatment model**: T ~ 1 (intercept only)
- **Outcome model**: Y ~ T + 1 (treatment + intercept)

The algorithm can still detect violations of unconfoundedness across multiple environments by testing for independence between these mechanism parameters.

## Backward Compatibility

All changes are backward compatible. Existing code using MINT with non-empty covariates will work exactly as before.

## Related Issues

- Fixes #12: Implement edge case with empty covariate set for MINT
- Helps with #7: Provides clearer error message for the "Need at least one array to concatenate" issue
