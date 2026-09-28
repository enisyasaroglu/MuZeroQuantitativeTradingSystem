from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.agents.muzero.mcts import (MinMaxStats, Node, run_mcts, ucb_score,
                           select_child, backpropagate)


def make_cfg(**kw):
    base = dict(action_space_dim=3, num_simulations=1, discount_factor=0.99,
                pb_c_base=19652, pb_c_init=1.25,
                root_dirichlet_alpha=0.3, root_exploration_fraction=0.25)
    base.update(kw)
    return SimpleNamespace(**base)


class FakeNetwork:
    device = "cpu"

    def __init__(self, logits=(0., 0., 0.), value=0.5, rewards=(0.01, 0.01, 0.01)):
        self.logits, self.v, self.rewards = torch.tensor([logits]), value, rewards

    def prediction(self, state):
        return self.logits, torch.tensor([[self.v]])

    def dynamics(self, state, action):
        return state + 1, torch.tensor([[self.rewards[int(action.item())]]])


def path(*rewards):
    """root -> n1 -> n2 ...; node i receives rewards[i-1]; root reward is 0."""
    nodes = [Node(0.0)] + [Node(0.5) for _ in rewards]
    for n, r in zip(nodes[1:], rewards):
        n.reward = r
    return nodes


def child(reward, visits, value, prior):
    c = Node(prior)
    c.reward, c.visit_count, c.value_sum = reward, visits, value * visits
    return c


# ---- backup / values (expected: green) ----
def test_backprop_single_edge():
    root, ch = nodes = path(0.01)
    backpropagate(nodes, 0.50, 0.99, MinMaxStats())
    assert ch.value() == pytest.approx(0.50)
    assert root.value() == pytest.approx(0.505)          # 0.01 + 0.99*0.50
    assert root.visit_count == ch.visit_count == 1

def test_backprop_three_deep():
    root, a, b, c = nodes = path(1.0, 2.0, 3.0)
    backpropagate(nodes, 10.0, 0.5, MinMaxStats())
    assert c.value() == pytest.approx(10.0)
    assert b.value() == pytest.approx(8.0)               # 3 + 0.5*10
    assert a.value() == pytest.approx(6.0)               # 2 + 0.5*8
    assert root.value() == pytest.approx(4.0)            # 1 + 0.5*6
    assert root.value() == pytest.approx(1 + 0.5*2 + 0.25*3 + 0.125*10)  # closed form

def test_backprop_averages_over_visits():
    root, ch = nodes = path(0.1)
    for v in (1.0, 2.0):
        backpropagate(nodes, v, 0.9, MinMaxStats())
    assert ch.visit_count == root.visit_count == 2
    assert ch.value() == pytest.approx(1.5)              # (1.0 + 2.0) / 2
    assert root.value() == pytest.approx(1.45)           # ((0.1+0.9) + (0.1+1.8)) / 2


# ---- MinMaxStats (expected: green) ----
def test_minmax_normalises_to_unit_interval():
    s = MinMaxStats()
    for v in (2.0, 6.0, 4.0):
        s.update(v)
    assert (s.normalize(2.0), s.normalize(6.0)) == (0.0, 1.0)
    assert s.normalize(4.0) == pytest.approx(0.5)
    assert MinMaxStats(SimpleNamespace(min=-1.0, max=1.0)).normalize(0.0) == pytest.approx(0.5)


# ---- selection (green) ----
def test_select_child_follows_prior_when_nothing_visited():
    parent = Node(0.0); parent.visit_count = 4
    parent.children = {0: Node(0.1), 1: Node(0.7), 2: Node(0.2)}
    action, _ = select_child(make_cfg(), parent, MinMaxStats())
    assert action == 1


# ---- the suspects (expected: RED today) ----
def test_ucb_prefers_child_with_higher_immediate_reward():
    cfg = make_cfg(discount_factor=0.9)
    parent = Node(0.0); parent.visit_count = 4
    a = child(reward=0.0, visits=1, value=1.0, prior=0.5)   # Q = 0.0 + 0.9*1.0 = 0.9
    b = child(reward=1.0, visits=1, value=1.0, prior=0.5)   # Q = 1.0 + 0.9*1.0 = 1.9
    stats = MinMaxStats(SimpleNamespace(min=0.0, max=3.0))
    gap = ucb_score(cfg, parent, b, stats) - ucb_score(cfg, parent, a, stats)
    assert gap == pytest.approx(1.0 / 3.0)                  # (1.9 - 0.9) / 3

def test_ucb_unvisited_child_gets_zero_value_term():
    parent = Node(0.0); parent.visit_count = 5
    fresh = Node(0.0)                                       # prior 0 => exploration term is 0
    stats = MinMaxStats(SimpleNamespace(min=-1.0, max=1.0))
    assert ucb_score(make_cfg(), parent, fresh, stats) == pytest.approx(0.0)

def test_backprop_feeds_minmax_with_q_values():
    nodes = path(0.1)
    stats = MinMaxStats()
    backpropagate(nodes, 2.0, 0.9, stats)
    assert stats.maximum == pytest.approx(1.9)              # child Q = 0.1 + 0.9*2.0, not V = 2.0


# ---- whole search with the fake network (green) ----
def test_run_mcts_single_simulation_matches_hand_calc():
    root = run_mcts(make_cfg(num_simulations=1), torch.zeros(1, 1), FakeNetwork(),
                    MinMaxStats(), add_exploration_noise=False)
    assert root.visit_count == 1
    assert sum(c.visit_count for c in root.children.values()) == 1
    assert root.value() == pytest.approx(0.01 + 0.99 * 0.50)   # 0.505

def test_run_mcts_visit_count_invariants():
    root = run_mcts(make_cfg(num_simulations=25), torch.zeros(1, 1),
                    FakeNetwork(logits=(0.1, 0.2, 0.3)), MinMaxStats(), add_exploration_noise=False)
    assert root.visit_count == 25
    assert sum(c.visit_count for c in root.children.values()) == 25
    assert np.isfinite(root.value())

def test_root_priors_equal_softmax_without_noise():
    net = FakeNetwork(logits=(0.1, 0.2, 0.3))
    root = run_mcts(make_cfg(), torch.zeros(1, 1), net, MinMaxStats(), add_exploration_noise=False)
    expected = torch.softmax(net.logits, dim=1).squeeze(0).numpy()
    np.testing.assert_allclose([root.children[a].prior for a in range(3)], expected, rtol=1e-5)