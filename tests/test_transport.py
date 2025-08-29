import numpy as np

from causal_falsify.algorithms.transport import TransportabilityTest


def make_fake_data(n_samples=200, n_covariates=3):
    rng = np.random.default_rng(123)
    outcome = rng.normal(size=(n_samples, 1))
    source = rng.binomial(1, 0.5, size=(n_samples, 1))
    covariates = rng.normal(size=(n_samples, n_covariates))
    treatment = rng.binomial(1, 0.5, size=(n_samples, 1))
    return outcome, source, covariates, treatment


def test_subsample_data_preserves_shape_and_limits():
    max_sample_size = 50
    tester = TransportabilityTest(max_sample_size=max_sample_size, seed=42)

    outcome, source, covariates, treatment = make_fake_data()

    outcome_sub, source_sub, covariates_sub, treatment_sub = tester.subsample_data(
        outcome, source, covariates, treatment
    )

    # 1. Max sample size respected
    assert outcome_sub.shape[0] <= max_sample_size
    assert source_sub.shape[0] == outcome_sub.shape[0]
    assert covariates_sub.shape[0] == outcome_sub.shape[0]
    assert treatment_sub.shape[0] == outcome_sub.shape[0]

    # 2. Shape consistency
    assert outcome_sub.shape[1] == outcome.shape[1]
    assert covariates_sub.shape[1] == covariates.shape[1]
    assert treatment_sub.shape[1] == treatment.shape[1]

    # 3. Source proportions roughly preserved
    orig_props = np.mean(source == 1)
    sub_props = np.mean(source_sub == 1)
    assert np.isclose(sub_props, orig_props, atol=0.1)


def test_subsample_reproducibility_with_seed():
    data = make_fake_data()

    tester1 = TransportabilityTest(max_sample_size=50, seed=123)
    tester2 = TransportabilityTest(max_sample_size=50, seed=123)

    subs1 = tester1.subsample_data(*data)
    subs2 = tester2.subsample_data(*data)

    # Same seed => identical subsamples
    for a, b in zip(subs1, subs2):
        assert np.array_equal(a, b)


def test_subsample_different_seed_changes_output():
    data = make_fake_data()

    tester1 = TransportabilityTest(max_sample_size=50, seed=123)
    tester2 = TransportabilityTest(max_sample_size=50, seed=456)

    subs1 = tester1.subsample_data(*data)
    subs2 = tester2.subsample_data(*data)

    # Different seeds => at least one array differs
    assert any(not np.array_equal(a, b) for a, b in zip(subs1, subs2))
