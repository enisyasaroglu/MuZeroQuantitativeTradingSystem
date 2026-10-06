import random
import pytest
from src.utils.replay_buffer import ReplayBuffer  # adjust to your real path


def make_game(length, reward=0.1, value=None):
    # actions are 1-indexed so the padding sentinel (0) never collides with a real position
    actions = [i + 1 for i in range(length)]
    g = {
        'obs': [f"obs_{i}" for i in range(length)],
        'actions': actions,
        'rewards': [reward] * length,
        'policies': [[0.2, 0.3, 0.5]] * length,
    }
    if value is not None:
        g['values'] = [value] * length
    return g


def test_last_transition_of_episode_is_reachable(monkeypatch):
    """Every position 0..game_len-1 must be reachable as current_idx for some
    start_index. Today position game_len-1 never is."""
    buf = ReplayBuffer(capacity=10, batch_size=1, unroll_steps=3, discount=0.9, td_steps=5)
    game_len = 6
    buf.save_game(make_game(game_len, value=0.0))

    seen = set()
    for forced in range(game_len):
        monkeypatch.setattr(random, "randint", lambda a, b, s=forced: min(s, b))
        batch = buf.sample_batch()
        seen.update(a - 1 for a in batch['actions'][0] if a != 0)

    assert seen == set(range(game_len)), f"never sampled: {set(range(game_len)) - seen}"


def test_nstep_return_matches_hand_calc(monkeypatch):
    buf = ReplayBuffer(capacity=10, batch_size=1, unroll_steps=1, discount=0.5, td_steps=2)
    buf.save_game(make_game(length=20, reward=1.0, value=100.0))
    monkeypatch.setattr(random, "randint", lambda a, b: 5)   # force a safe mid-episode start
    batch = buf.sample_batch()
    assert batch['target_values'][0][0] == pytest.approx(1.0 + 0.5 * 1.0 + 0.25 * 100.0)


def test_nstep_return_has_no_bootstrap_past_episode_end(monkeypatch):
    buf = ReplayBuffer(capacity=10, batch_size=1, unroll_steps=1, discount=0.9, td_steps=5)
    buf.save_game(make_game(length=6, reward=1.0, value=1000.0))  # value huge and easy to spot if it leaks
    monkeypatch.setattr(random, "randint", lambda a, b: b)        # force the last valid start_index
    batch = buf.sample_batch()
    assert batch['target_values'][0][0] < 10