import jax.numpy as jnp

from perpetua2.utils.metrics import compute_aic


def test_compute_aic_basic():
    """Test basic AIC computation: AIC = -2*log_likelihood + 2*k."""
    log_likelihood = -10.0
    k = 3
    expected_aic = 26.0  # -2*(-10) + 2*3 = 20 + 6

    result = compute_aic(log_likelihood, k)

    assert jnp.isclose(result, expected_aic)


def test_compute_aic_edge_cases():
    """Test edge cases: zero values."""
    # Zero log-likelihood
    assert jnp.isclose(compute_aic(0.0, 2), 4.0)  # 0 + 4

    # Zero parameters
    assert jnp.isclose(compute_aic(-5.0, 0), 10.0)  # 10 + 0


def test_compute_aic_formula_verification():
    """Test the AIC formula with multiple cases."""
    test_cases = [
        (-15.7, 4, 39.4),   # typical case
        (2.3, 1, -2.6),     # positive log-likelihood
    ]

    for log_likelihood, k, expected in test_cases:
        result = compute_aic(log_likelihood, k)
        assert jnp.isclose(result, expected, atol=1e-6)
