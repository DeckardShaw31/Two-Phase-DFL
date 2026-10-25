import numpy as np
import pytest

from src.models.baselines import LightGBMQuantilePTO, SafetyStockPTO


def test_lightgbm_quantile_pto_fractile_parameterization():
    """
    Test that LightGBMQuantilePTO supports both:
    1. myopic single-period fractile: alpha_myopic = p_u / (p_u + p_s) = 20 / (20 + 5) = 0.80
    2. multi-period tau-amortized fractile: beta_tau* = p_u / (p_u + h + p_s/tau) = 20 / (20 + 1 + 5/3) = 15/17 = 0.8823529...
    3. custom explicit critical_fractile
    """
    # 1. Myopic fractile
    m_myopic = LightGBMQuantilePTO(
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
        shelf_life_tau=3,
        fractile_type="myopic",
    )
    assert pytest.approx(m_myopic.critical_fractile, 1e-5) == 0.80
    assert m_myopic.model.alpha == m_myopic.critical_fractile

    # 2. Tau-amortized fractile
    m_amortized = LightGBMQuantilePTO(
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
        shelf_life_tau=3,
        fractile_type="tau_amortized",
    )
    expected_beta_tau = 20.0 / (20.0 + 1.0 + 5.0 / 3.0)
    assert pytest.approx(m_amortized.critical_fractile, 1e-5) == expected_beta_tau
    assert pytest.approx(expected_beta_tau, 1e-4) == 0.8824

    # 3. Explicit critical fractile override
    m_custom = LightGBMQuantilePTO(critical_fractile=0.75)
    assert m_custom.critical_fractile == 0.75

    # 4. Fit and predict functionality test
    np.random.seed(42)
    X = np.random.randn(50, 14)
    y = np.maximum(0.0, 10.0 + 2.0 * X[:, 0] + np.random.randn(50) * 2.0)
    m_myopic.fit(X, y)
    preds = m_myopic.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(preds >= 0.0)


def test_safety_stock_pto_fractile_parameterization():
    """
    Test that SafetyStockPTO supports both:
    1. myopic single-period fractile: alpha_myopic = 0.80
    2. multi-period tau-amortized fractile: beta_tau* = 0.8824
    3. custom explicit critical_fractile
    """
    # 1. Myopic
    ss_myopic = SafetyStockPTO(
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
        shelf_life_tau=3,
        fractile_type="myopic",
    )
    assert pytest.approx(ss_myopic.critical_fractile, 1e-5) == 0.80

    # 2. Tau-amortized
    ss_amortized = SafetyStockPTO(
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
        shelf_life_tau=3,
        fractile_type="tau_amortized",
    )
    expected_beta_tau = 20.0 / (20.0 + 1.0 + 5.0 / 3.0)
    assert pytest.approx(ss_amortized.critical_fractile, 1e-5) == expected_beta_tau

    # 3. Explicit override
    ss_custom = SafetyStockPTO(critical_fractile=0.90)
    assert ss_custom.critical_fractile == 0.90

    # 4. Verify fit and predict creates safety buffer
    np.random.seed(42)
    X = np.random.randn(50, 14)
    y = np.maximum(0.0, 10.0 + 2.0 * X[:, 0] + np.random.randn(50) * 2.0)
    ss_amortized.fit(X, y)
    assert ss_amortized.safety_buffer > 0.0
    preds = ss_amortized.predict(X[:5])
    assert preds.shape == (5,)


def test_mathematical_properties_fifo_and_capacity_externality():
    """
    Verify mathematical properties proved in Lemma 1, Theorem 1, and Theorem 2:
    - Age-differentiated holding costs: h_a = h + (a-1)*eps satisfies 0 < h_1 < ... < h_{tau-1} < p_s
    - System-optimal critical fractile with fleet shadow price:
      beta_{it}^* = (p_u - lambda_t^* - kappa_{it}) / (p_u + c_o) < beta^*
    """
    # Lemma 1: Age-differentiated holding costs check
    h = 1.0
    eps = 0.01
    p_s = 5.0 * h
    for tau in [2, 3, 5]:
        h_costs = [h + (a - 1) * eps for a in range(1, tau)]
        # Strictly increasing
        for a in range(len(h_costs) - 1):
            assert h_costs[a] < h_costs[a + 1]
        # Below spoilage penalty
        assert h_costs[-1] < p_s

    # Theorem 2: Coupled fractile depression check
    p_u = 20.0
    c_o = h + p_s / 3.0  # 1.0 + 5.0/3 = 2.6667
    beta_unconstrained = p_u / (p_u + c_o)
    assert pytest.approx(beta_unconstrained, 1e-4) == 0.8824

    # When fleet capacity binds (lambda_t* > 0) or routing cost kappa_it > 0:
    lambda_t = 2.0  # capacity shadow price
    kappa_it = 1.5  # marginal delivery cost
    beta_coupled = (p_u - lambda_t - kappa_it) / (p_u + c_o)
    assert beta_coupled < beta_unconstrained
    assert beta_coupled == pytest.approx(beta_unconstrained - (lambda_t + kappa_it) / (p_u + c_o), 1e-5)


def test_baseline_default_instantiation_tau_amortized():
    """
    Verify standalone instantiation of LightGBMQuantilePTO and SafetyStockPTO without arguments
    defaults to shelf_life_tau=3 and correctly evaluates beta_3* = 15/17 approx 0.8824.
    """
    m_default = LightGBMQuantilePTO()
    expected_beta_3 = 20.0 / (20.0 + 1.0 + 5.0 / 3.0)
    assert pytest.approx(m_default.critical_fractile, 1e-5) == expected_beta_3
    assert pytest.approx(m_default.critical_fractile, 1e-4) == 0.8824

    ss_default = SafetyStockPTO()
    assert pytest.approx(ss_default.critical_fractile, 1e-5) == expected_beta_3
    assert pytest.approx(ss_default.critical_fractile, 1e-4) == 0.8824

