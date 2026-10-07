# MuZero-Style Trading Agent

A research project that asks a simple question: can an agent that plans ahead trade better than one that only reacts?

I built it as my final-year project at the University of Greenwich. It adapts DeepMind's MuZero to daily SPY trading and compares it with a PPO baseline and three simple benchmarks. The project is still being developed, and this page tries to be clear about what works, what is being redone, and what I don't know yet.

## Project status

I found and fixed two bugs in the planning code in October 2026:

- The tree search ignored the immediate reward when choosing between actions.
- The value targets used only the next five rewards and left out the bootstrap term.

All models trained before these fixes are out of date, and the tests that caught the bugs now pass. I am retraining and rebuilding the evaluation so that every result can be traced to a configuration, a data file and a checkpoint. Results from the earlier version of this work cannot currently be reproduced from this repository, so I don't quote them here.

## What the project explores

A reactive agent (PPO) maps what it sees today to an action. A planning agent (MuZero) first learns a small model of how the market moves, then uses that model to imagine what each action might lead to before choosing one.

Board games give MuZero a clean setting. Markets don't, because prices are noisy and the rules change. So the interesting question is whether planning helps at all in this setting, and the honest answer will include cases where it doesn't.

## How it works

```
Yahoo Finance data -> features and splits -> trading environment -> agent -> evaluation
```

**Data.** Daily SPY prices from Yahoo Finance, January 2010 to December 2025. Training uses 2010 to March 2021, validation uses June 2021 to October 2023, and test uses January 2024 to December 2025, with a 60-day gap between splits. The first 60 rows of each split serve only as lookback, so trading starts a little later.

**Environment.** The agent sees the previous 60 days and chooses Short, Neutral or Long. It earns that position multiplied by the next day's return, minus a 0.1% fee for each unit of position change. Switching from Short to Long therefore costs 0.2%. The agent decides after one close and earns the following day's move, but trades at the same close it just observed, which assumes a market-on-close order.

**MuZero agent.** Three networks work together: one encodes the last 60 days into a latent state, one predicts the next state and reward for a chosen action, and one outputs a policy and a value. A tree search over the learned model scores each action as its predicted reward plus the discounted value of where it leads.

**PPO baseline.** A standard actor-critic agent with its own LSTM encoder, trained in the same environment.

**Benchmarks.** Buy-and-hold, always-cash, and a 50-day moving average (long when the previous close was above its average).

## Guarding against look-ahead bias

Backtests are easy to get wrong in ways that flatter the results, so these are the safeguards in place:

- Splits are chronological, with a 60-day gap between them.
- Normalisation statistics come from the training split only.
- Indicators measured in price units are divided by the close, so they stay on the same scale as the index rises.
- The observation window ends the day before the return it is paid on.

Still to add: a test that proves features at time *t* don't change when later data is appended, and saving the normalisation statistics alongside each model.

## Training and evaluation

Each training episode is a random 252-day window of the training data, played with exploration switched on. Training uses raw net returns as the reward. A Differential Sharpe Ratio reward is implemented and tested, but the current runs don't use it.

Evaluation plays a full split in order with exploration off: MuZero with temperature 0, PPO taking its most likely action. Each run records the dates traded, the data and checkpoint fingerprints, and the configuration used, and every evaluation is appended to `logs/evaluation/evaluation_log.csv`. The validation split is for choosing models, and the test split is reported once.

## Results

I'm not publishing numbers yet. Three things need to happen first: a full retraining with the fixed code on the 2010 to 2025 data, several random seeds, and one clean evaluation of the test split on a model chosen beforehand using validation only.

One finding worth recording: earlier checkpoints, trained before the search fix, held cash on every single day when evaluated. The retraining is testing whether that behaviour was caused by the bugs or whether a cash-only policy is simply what the agent learns when the market offers little to exploit.

## Feature status

| Area | Status | Notes |
|---|---|---|
| Data pipeline | Implemented, tested | Leakage tests being strengthened |
| Single-asset environment | Implemented, tested | |
| Differential Sharpe reward | Implemented, tested | Not used in current runs |
| MuZero agent | Implemented, tested | Retraining after the October fixes |
| PPO baseline | Implemented, lightly tested | Window-end handling to be fixed |
| Evaluation and benchmarks | Implemented | Rewritten in October |
| Multi-asset environment | Experimental | No evaluation path yet |
| Portfolio optimisation methods | Implemented, tested | Not yet reviewed |
| Risk manager | Partial | |
| Regime detection | Experimental | |
| Backtesting | Partial | Cost models only so far |
| Order execution and brokers | Placeholder | |
| Docker and Terraform | Not yet reviewed | |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download data and build the train, validation and test files
python3 src/pipeline/processor.py

# 2. Train. Each run writes its models and a configuration record to its folder.
python3 main_muzero.py --run-dir runs/muzero_001
python3 main_ppo.py --run-dir runs/ppo_001

# 3. Evaluate on the validation split (baselines are always included)
python3 evaluate.py --agent all --split val \
    --muzero-ckpt runs/muzero_001/muzero_checkpoint_500.pth \
    --ppo-ckpt runs/ppo_001/ppo_final.pth

# 4. Run the tests
pytest
```

A quick check that everything runs: `python3 main_muzero.py --episodes 12 --simulations 20 --run-dir runs/smoke_test`.

## Repository layout

```
configs/      Settings for data, MuZero and PPO
src/pipeline/ Data download, validation, features and splits
src/env/      Trading environments and the Differential Sharpe reward
src/networks/ Neural network building blocks
src/agents/   MuZero (search and training) and PPO
src/utils/    Metrics, replay buffer, logging
src/portfolio, src/risk, src/regime, src/strategies, src/backtesting
              Portfolio, risk, regime and benchmark modules
tests/        Automated tests
notebooks/    Exploration and sanity checks
```

## Testing

150 tests run in a few seconds and use synthetic data, so no download or trained model is needed. They cover the environment, the reward, search mechanics, the replay buffer, the metrics and the portfolio methods. Two of the most useful ones check that the search prefers an action with a higher immediate reward and that value targets add the correct bootstrap term. Both failed before the October fixes.

## Limitations

- One asset, one data window, one random seed so far. Differences between agents are not statistically meaningful until this changes.
- The validation and test windows are each about two years long, so a Sharpe ratio measured on either is still uncertain by roughly 0.7 to 0.8.
- Costs are a flat 0.1% per unit of position change, with no slippage, no cost for borrowing shares to short, and no interest on cash.
- The search uses 150 simulations per decision, far fewer than the original MuZero paper's 800.
- The PPO baseline isn't compute-matched to MuZero, so comparisons between them need care.
- Daily data and same-close execution suit research. They are not a model of live trading.

## Roadmap

Roughly in order:

1. Retrain with the fixed code. Run several seeds, with validation-based model selection.
2. Report confidence intervals for every result.
3. Strengthen the leakage tests and save normalisation statistics with each model.
4. Handle window ends correctly in PPO.
5. A full backtesting engine with richer transaction costs.
6. Regime detection and multi-asset portfolios.

## Companion project

A second project applies swarm optimisation algorithms to constrained portfolio selection. It lives in its own repository and is the natural partner to this one: this project decides direction for one asset, and that one decides how to split capital across assets.

## References

- Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model, Nature.
- Moody and Wu (1997), Optimization of trading systems and portfolios.
- Vittori et al. (2021), Monte Carlo Tree Search for Trading and Hedging, ICAIF.

## License

MIT