"""
Small scheduling utilities for training self-play.
"""


def temperature_schedule(episode: int, num_episodes: int) -> float:
    """
    Standard staged temperature annealing for MuZero/AlphaZero-style
    self-play action selection (see MuZeroAgent.select_action).

    High temperature early (full exploration -- actions are sampled
    roughly proportional to MCTS visit counts, not always the current
    argmax) prevents the self-reinforcing policy collapse that results
    from always playing the greedy action from the start of training:
    with a randomly-initialised network, whichever action happens to get
    a slightly higher visit count first would otherwise be played every
    single step, filling the replay buffer with degenerate, all-one-
    action trajectories that reinforce themselves with every training
    step. Annealing toward a lower temperature later in training lets the
    policy sharpen once it has had a chance to explore.

    Stage boundaries and specific temperature values are a standard,
    simple staged schedule (similar in spirit to the original AlphaZero
    paper's temperature drop after move 30) -- not tuned/optimised here,
    just enough to prevent early collapse.
    """
    if num_episodes <= 0:
        return 1.0
    frac = episode / num_episodes
    if frac < 0.5:
        return 1.0
    elif frac < 0.75:
        return 0.5
    else:
        return 0.25


def entropy_coef_schedule(episode: int, num_episodes: int, start: float = 0.05, end: float = 0.02) -> float:
    """
    Linear entropy-bonus annealing for PPO training.

    A static, low entropy coefficient (0.01) was observed to let PPO's
    policy collapse to always sampling a single action after ~500
    episodes of training on real data (confirmed by zero deviation from
    "Long" across an entire 182-step evaluation episode -- statistically
    near-impossible if the policy retained even 1% probability mass on
    other actions). Starting with a higher entropy bonus and annealing it
    down over training keeps exploration pressure strong early (when the
    policy is most prone to prematurely committing to one action) while
    still letting the policy sharpen later once it has had a chance to
    explore -- the same principle as temperature_schedule above, applied
    to PPO's entropy bonus instead of MCTS action sampling.

    end=0.02 rather than 0.01: a full 500-episode run on real data
    (evaluate.py --split test, deterministic policy) reproduced this
    exact collapse -- PPO's action distribution matched Buy-and-Hold's
    forced all-Long strategy exactly (0 position changes, every financial
    metric identical to the decimal). The schedule was annealing straight
    down to the same value already known to cause it. 0.02 keeps
    meaningful downward annealing pressure while staying above the
    confirmed failure point; needs re-verification with a full retrain,
    not assumed fixed until then.
    """
    if num_episodes <= 0:
        return start
    frac = min(episode / num_episodes, 1.0)
    return start + (end - start) * frac