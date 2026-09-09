import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.constraints import PortfolioConstraints


def test_default_matches_pso_original_behaviour():
    c = PortfolioConstraints()
    w = c.project(np.array([0.5, -0.2, 0.3, 0.9]))
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-9


def test_bounds_are_enforced():
    c = PortfolioConstraints(min_weight=0.0, max_weight=0.5)
    w = c.project(np.array([0.9, 0.05, 0.05]))
    assert np.all(w <= 0.5 + 1e-9)


def test_cardinality_keeps_only_top_k():
    c = PortfolioConstraints(max_assets=2)
    w = c.project(np.array([0.1, 0.6, 0.05, 0.25]))
    assert np.sum(w > 1e-9) == 2
    assert abs(w.sum() - 1.0) < 1e-9


def test_degenerate_all_clipped_falls_back_to_equal_weight():
    c = PortfolioConstraints()
    w = c.project(np.array([-1.0, -2.0, -0.5]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, 1.0 / 3)


def test_is_feasible_true_for_valid_weights():
    assert PortfolioConstraints().is_feasible(np.array([0.3, 0.3, 0.4]))


def test_is_feasible_false_when_over_max_weight():
    assert not PortfolioConstraints(max_weight=0.5).is_feasible(np.array([0.8, 0.1, 0.1]))


def test_is_feasible_false_when_cardinality_exceeded():
    assert not PortfolioConstraints(max_assets=2).is_feasible(np.array([0.34, 0.33, 0.33]))


def test_invalid_bounds_raise():
    try:
        PortfolioConstraints(min_weight=0.6, max_weight=0.4)
        assert False, "expected ValueError"
    except ValueError:
        pass