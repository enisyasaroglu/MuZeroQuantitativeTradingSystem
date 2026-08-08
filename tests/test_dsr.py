import random
import numpy as np
from src.env.rewards import DifferentialSharpeRatio


def test_warmup_returns_zero():
    """Verify dsr.step() returns 0.0 during the warmup window.

    Variance calculations are unreliable before enough steps accumulate,
    so early rewards must remain zero.
    """
    dsr = DifferentialSharpeRatio(warmup_steps=5)
    for _ in range(5):
        reward = dsr.step(0.01)
        assert reward == 0.0, f"Expected warmup reward 0.0, got {reward}"
    print("test_warmup_returns_zero PASSED")


def test_reset_restores_zero_state():
    """Verify calling dsr.reset() restores tracking variables back to zero."""
    dsr = DifferentialSharpeRatio(warmup_steps=2)
    for r in [0.01, -0.02, 0.03, 0.04, -0.01]:
        dsr.step(r)

    assert dsr.A != 0.0 or dsr.B != 0.0, "Expected non-zero state before reset"

    dsr.reset()
    assert dsr.A == 0.0, f"Expected dsr.A to be 0.0, got {dsr.A}"
    assert dsr.B == 0.0, f"Expected dsr.B to be 0.0, got {dsr.B}"
    print("test_reset_restores_zero_state PASSED")


def test_clip_bounds_extreme_input():
    """Verify reward clipping keeps outputs strictly within configured bounds."""
    dsr = DifferentialSharpeRatio(warmup_steps=2, clip=10.0)
    for _ in range(3):
        dsr.step(0.001)

    reward_pos = dsr.step(1000.0)
    assert -10.0 <= reward_pos <= 10.0, f"Positive reward out of bounds: {reward_pos}"

    reward_neg = dsr.step(-1000.0)
    assert -10.0 <= reward_neg <= 10.0, f"Negative reward out of bounds: {reward_neg}"
    print("test_clip_bounds_extreme_input PASSED")


def test_single_call_per_step_is_deterministic():
    """Verify state updates match exact exponential moving average math.

    Ensures single step updates progress predictably without double-updating
    statistics.
    """
    dsr = DifferentialSharpeRatio(eta=0.1, warmup_steps=0, clip=1e9)
    returns = [0.01, 0.02, -0.01, 0.015, -0.005]
    for r in returns:
        dsr.step(r)

    # Recompute theoretical exponential moving averages
    A, B = 0.0, 0.0
    for r in returns:
        dA = r - A
        dB = (r**2) - B
        A += 0.1 * dA
        B += 0.1 * dB

    assert abs(dsr.A - A) < 1e-10, f"Mismatch in A: {dsr.A} vs {A}"
    assert abs(dsr.B - B) < 1e-10, f"Mismatch in B: {dsr.B} vs {B}"
    print("test_single_call_per_step_is_deterministic PASSED")


def test_reward_responds_to_return_quality():
    """Verify DSR gives higher total reward to smooth returns over volatile ones.

    Tests that risk-adjusted performance is rewarded over high-variance streams
    with equal mean returns.
    """
    np.random.seed(0)
    dsr_smooth = DifferentialSharpeRatio(warmup_steps=5)
    dsr_volatile = DifferentialSharpeRatio(warmup_steps=5)

    smooth_returns = np.full(50, 0.005)
    volatile_returns = np.random.choice([0.05, -0.04], size=50) + 0.005

    smooth_rewards = [dsr_smooth.step(r) for r in smooth_returns]
    volatile_rewards = [dsr_volatile.step(r) for r in volatile_returns]

    assert sum(smooth_rewards[5:]) > sum(volatile_rewards[5:]), (
        "DSR should favor smoother return streams over volatile ones"
    )
    print("test_reward_responds_to_return_quality PASSED")


if __name__ == "__main__":
    test_warmup_returns_zero()
    test_reset_restores_zero_state()
    test_clip_bounds_extreme_input()
    test_single_call_per_step_is_deterministic()
    test_reward_responds_to_return_quality()
    print("\nALL DSR SANITY CHECKS PASSED")