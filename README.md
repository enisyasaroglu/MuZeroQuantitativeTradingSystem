# MuZero-Inspired Quantitative Trading Agent

A model-based reinforcement learning trading agent that adapts DeepMind's **MuZero** architecture — built for perfect-information games like Chess and Go — to the imperfect, stochastic world of financial markets.

> **Project Overview:** 
Standard RL trading agents (PPO, DQN) react to the market but can't "plan ahead." Planning algorithms like AlphaZero *can* look ahead, but only because they assume access to a perfect simulator of what happens next — something that doesn't exist in finance. This project builds a MuZero agent that **learns its own internal model of market dynamics** and plans inside that learned model using a custom **Open-Loop Monte Carlo Tree Search**, adapted specifically to handle the fact that one trading action can lead to many different possible future outcomes, not one.

---

## Technical Challenges & Solutions

| Problem | Standard approach fails because... | This project's answer |
|---|---|---|
| No known "rules" for how prices move | AlphaZero requires a perfect simulator (game rules) | MuZero learns an approximate dynamics model instead |
| Price transitions are random, not deterministic | Standard MCTS assumes one action → one exact next state | Open-Loop MCTS: tree nodes represent *action sequences*, not states |
| Historical data can leak into the future | Naive train/test splits and indicator computation leak information | 60-day embargo gaps, per-split indicator computation, train-only normalisation |
| Raw returns reward reckless risk-taking | Optimising for profit alone encourages excessive volatility | Differential Sharpe Ratio used as a dense, risk-adjusted reward signal |

---

## Headline Results (Out-of-Sample Test Set, 2023–2024)

| Agent | Total Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| **MuZero** | -9.95% | -0.88 | -20.6% |
| PPO (baseline) | -25.21% | -2.60 | -26.0% |
| Buy-and-Hold | +27.10% | +1.89 | -8.4% |

**MuZero consistently outperformed the PPO baseline** across both out-of-sample evaluation periods (+29.5 pts validation, +15.3 pts test). **Neither RL agent beat passive Buy-and-Hold** during the exceptionally strong 2023–2024 bull market — MuZero had correctly learned a defensive, short-biased strategy during the 2022 bear market and failed to "unlearn" that bias once the regime shifted. This is documented and analysed as a finding, not hidden as a failure, see [Limitations](#limitations).

---

## Architecture

Six-layer system: **Data Pipeline → Trading Environment → Shared Networks → Agents (MuZero / PPO) → Training Loop → Evaluation**. Full diagram in `docs/architecture.pdf`.

- **Representation Network** — 2-layer LSTM, encodes a 60-day lookback window into a 64-dim latent state
- **Dynamics Network** — the learned "world model"; predicts next latent state + reward from a hidden state and action
- **Prediction Network** — policy + value heads over the latent state
- **Open-Loop MCTS** — UCB-based tree search over action sequences, using the exact constants (`pb_c_base=19652`, `pb_c_init=1.25`) from Schrittwieser et al. (2020)
- **PPO baseline** — identical LSTM encoder to MuZero, so the *only* experimental variable is the presence of planning

---

## Engineering highlights

This wasn't just "run the algorithm and report numbers." Three environment bugs and one training-loop bug were found and fixed via a 10-file test suite during development:

- **Portfolio reset bug** — episodes were silently inheriting the previous episode's ending capital, corrupting validation comparisons (one checkpoint showed a false "£13.91" near-total-loss result caused entirely by this bug).
- **Reward signal double-counted** — the Differential Sharpe Ratio's internal EMA state was being advanced twice per step, corrupting the reward signal from step one of every episode.
- **Off-by-one reward misalignment** in the MuZero unrolled training loss — the reward head was being trained to predict the *next* step's reward instead of the current transition's, caught by isolating the reward-loss curve during integration testing.

Full detail in `docs/implementation_notes.md`.

---

## Tech Stack

`Python` · `PyTorch` · `Gymnasium` · `yfinance` · `stockstats` · `NumPy` / `Pandas` · `Matplotlib`

---

## Repository Structure

```
├── configs/ # Experiment configuration layer
│ ├── base_config.py # Shared hyperparameters
│ ├── muzero_config.py # MuZero-specific settings
│ └── ppo_config.py # PPO baseline settings
├── data/
│ ├── raw/ # Raw market data (OHLCV)
│ └── processed/ # Cleaned & engineered dataset
├── src/
│ ├── data_pipeline/ # Data engineering layer
│ │ ├── downloader.py # Yahoo / FinRL data fetching
│ │ ├── preprocessor.py # Indicators, normalization, splits
│ │ └── alternative_fetcher.py
│ ├── environment/ # Market simulation layer
│ │ ├── trading_env.py # Gym-style trading environment
│ │ └── rewards.py # Differential Sharpe Ratio reward
│ ├── networks/ # Deep learning models
│ │ ├── representation.py # State encoder (LSTM/GRU)
│ │ ├── dynamics.py # Transition model
│ │ ├── prediction.py # Policy + value heads
│ │ ├── muzero_networks.py # MuZero composite network
│ │ ├── ppo_networks.py # PPO actor-critic network
│ │ └── shared.py # Shared neural components
│ ├── agents/ # Decision-making agents
│ │ ├── muzero/ # MuZero implementation
│ │ └── baselines/ # PPO and other baselines
│ ├── planning/ # Search / decision planning
│ │ ├── mcts.py # Monte Carlo Tree Search
│ │ └── node.py # Tree node structure
│ ├── training/ # Learning pipeline
│ │ ├── trainer.py # Training loop (loss + optimisation)
│ │ ├── replay_buffer.py # Experience storage
│ │ └── evaluator.py # Backtesting + evaluation metrics
│ └── utils/ # Utility functions
│ ├── metrics.py # Sharpe, returns, drawdown
│ ├── rewards.py # Shared reward functions (if needed)
│ └── seeding.py # Reproducibility utilities
├── configs/ # Experiment configs (YAML/Python hybrid)
├── logs/
│ ├── checkpoints/ # Model weights during training
│ ├── tensorboard/ # Training logs
│ ├── models/ # Best saved models
│ │ ├── muzero_best_model.pth
│ │ └── ppo_best_model.pth
│ └── evaluation/ # Backtest results
│ ├── all_results.json
│ ├── test_results.csv
│ ├── validation_results.csv
│ ├── test_equity_curves.png
│ └── validation_equity_curves.png
├── notebooks/ # EDA + sanity checks
│ ├── 01_data_exploration.ipynb
│ └── 02_sine_wave_sanity_check.ipynb
├── tests/ # Unit + integration tests
├── main_muzero.py # MuZero training entry point
├── main_ppo.py # PPO baseline training
├── evaluate.py # Model evaluation script
├── README.md
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main_ppo.py --config configs/ppo_config.py       # train baseline
python main_muzero.py --config configs/muzero_config.py  # train MuZero
python evaluate.py --agent muzero --split test           # evaluate
pytest tests/                                            # run test suite
```

---

## Limitations

- **Single asset (SPY), single train/val/test split, single random seed.** The MuZero > PPO result is directionally consistent with the architecture's expected behaviour but is **not yet a statistically validated claim** — see Roadmap.
- **MCTS simulation budget of 50/step**, vs. 800 in the original MuZero paper — a single-CPU compute constraint, not a design choice.
- **Flat 0.1% transaction cost**, no slippage or market impact modelling.
- **No online regime detection** — the agent's biggest documented failure mode.

## Roadmap

- [ ] Multi-seed training with confidence intervals across multiple non-overlapping historical windows
- [ ] Regime-detection layer (HMM over latent states) to address the short-bias hysteresis problem
- [ ] Multi-asset portfolio extension — continuous position weights across SPY, QQQ, GLD, individual equities
- [ ] Realistic transaction cost model (slippage, market impact)
- [ ] Paper-trading integration via a broker API for live-data validation
- [ ] Cloud GPU scaling to raise MCTS simulation budget toward the original paper's 800/step

## Related Project

[**Swarm Intelligence for Portfolio Optimisation**](../swarm-intelligence): a companion mini-project applying Particle Swarm Optimisation to constrained Markowitz portfolio selection. The natural next step connecting both projects: use PSO (or a learned allocator) to size positions **across assets**, while MuZero decides direction **within** each asset — see Roadmap above.

## References

Schrittwieser et al. (2020), *Mastering Atari, Go, chess and shogi by planning with a learned model*, Nature.
Vittori et al. (2021), *Monte Carlo Tree Search for Trading and Hedging*, ICAIF '21.
Full bibliography in the accompanying dissertation.

## License

MIT# MuZeroQuantitativeTradingSystem
