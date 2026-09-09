import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.momentum_views import MomentumViewGenerator


def test_higher_relative_momentum_gets_positive_view():
    rng = np.random.RandomState(0)
    T = 60
    weak = rng.normal(0.0001, 0.01, T)
    strong = weak + 0.01
    flat = rng.normal(0.0, 0.01, T)
    returns = np.stack([weak, strong, flat], axis=1)

    P, Q, omega = MomentumViewGenerator().generate_views(returns)
    assert Q[1] > Q[0]


def test_zero_cross_sectional_dispersion_gives_zero_views():
    base = np.random.RandomState(1).normal(0.0, 0.01, 40)
    returns = np.stack([base, base, base], axis=1)
    P, Q, omega = MomentumViewGenerator().generate_views(returns)
    assert np.allclose(Q, 0.0)


def test_P_is_identity_matrix():
    returns = np.random.RandomState(2).normal(0.0005, 0.01, size=(30, 4))
    P, Q, omega = MomentumViewGenerator().generate_views(returns)
    assert np.allclose(P, np.eye(4))


def test_omega_is_diagonal_and_positive():
    returns = np.random.RandomState(3).normal(0.0005, 0.01, size=(30, 3))
    P, Q, omega = MomentumViewGenerator().generate_views(returns)
    assert np.allclose(omega, np.diag(np.diagonal(omega)))
    assert np.all(np.diagonal(omega) > 0)


def test_noisier_asset_gets_wider_omega_same_T():
    """RISK SCALING, checked directly: with the same number of
    observations, a MORE VOLATILE asset's view should be LESS confident
    (larger Omega diagonal entry) than a calmer asset's."""
    rng = np.random.RandomState(4)
    T = 50
    calm = rng.normal(0.0005, 0.005, T)
    volatile = rng.normal(0.0005, 0.03, T)
    returns = np.stack([calm, volatile], axis=1)

    P, Q, omega = MomentumViewGenerator().generate_views(returns)
    assert omega[1, 1] > omega[0, 0]


def test_raises_with_fewer_than_two_observations():
    returns = np.array([[0.01, 0.02, 0.03]])
    try:
        MomentumViewGenerator().generate_views(returns)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_hyperparameters_raise():
    for bad in (0, -1):
        try:
            MomentumViewGenerator(tilt_strength=bad)
            assert False
        except ValueError:
            pass
        try:
            MomentumViewGenerator(omega_scale=bad)
            assert False
        except ValueError:
            pass


def test_tilt_strength_scales_view_magnitude_linearly():
    returns = np.random.RandomState(5).normal(0.0005, 0.01, size=(40, 3))
    returns[:, 0] += 0.005

    _, Q_weak, _ = MomentumViewGenerator(tilt_strength=0.5).generate_views(returns)
    _, Q_strong, _ = MomentumViewGenerator(tilt_strength=2.0).generate_views(returns)
    assert np.allclose(Q_strong, Q_weak * 4.0, atol=1e-8)