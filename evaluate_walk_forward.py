import os
import glob
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

TEST_DATA_PATH = "data/processed/test_data.csv"
CHECKPOINT_DIR = "src/checkpoints"
INITIAL_CAPITAL = 100000.0
TRANSACTION_FEE = 0.0005


@dataclass
class PortfolioState:
    cash: float = INITIAL_CAPITAL
    shares: float = 0.0
    position_state: int = 0
    portfolio_value: float = INITIAL_CAPITAL


def extract_price_series(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    candidates = ["close", "Close", "CLOSE", "adj close", "price"]
    for col in candidates:
        if col in df.columns:
            return df[col].values, df

    if "log_return" in df.columns:
        synthetic_prices = 100.0 * np.exp(np.cumsum(df["log_return"].values))
        df_copy = df.copy()
        df_copy["synthetic_price"] = synthetic_prices
        return synthetic_prices, df_copy

    raise KeyError("No price or log_return column found in dataset.")


def execute_trade(action: int, current_price: float, state: PortfolioState, fee_rate: float) -> PortfolioState:
    target_position = action - 1
    
    if target_position == state.position_state:
        state.portfolio_value = state.cash + (state.shares * current_price)
        return state

    if state.shares != 0:
        liquidation_value = state.shares * current_price
        fee = abs(liquidation_value) * fee_rate
        state.cash += liquidation_value - fee
        state.shares = 0.0

    current_value = state.cash
    if target_position == 1:
        fee = current_value * fee_rate
        allocatable = current_value - fee
        state.shares = allocatable / current_price
        state.cash = 0.0
    elif target_position == -1:
        fee = current_value * fee_rate
        allocatable = current_value - fee
        state.shares = -allocatable / current_price
        state.cash = current_value * 2.0 - fee

    state.position_state = target_position
    state.portfolio_value = state.cash + (state.shares * current_price)
    return state


def evaluate_single_checkpoint(df: pd.DataFrame, prices: np.ndarray, model: Any) -> Dict[str, float]:
    state = PortfolioState()
    non_feature_cols = ["date", "timestamp", "Date", "Timestamp", "symbol", "Symbol", "tic", "synthetic_price"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    features = numeric_df.values

    if hasattr(model, "eval"):
        model.eval()

    actions = []
    values = []

    for t in range(len(df)):
        price = prices[t]
        feat_tensor = torch.FloatTensor(features[t]).unsqueeze(0)

        with torch.no_grad():
            if hasattr(model, "policy_net"):
                logits = model.policy_net(feat_tensor)
            elif hasattr(model, "forward_policy"):
                logits, _ = model.forward_policy(feat_tensor)
            elif callable(model):
                out = model(feat_tensor)
                logits = out[0] if isinstance(out, tuple) else out
            else:
                logits = torch.tensor([[0.33, 0.33, 0.34]])

            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            action = int(np.argmax(probs))

        actions.append(action)
        state = execute_trade(action, price, state, TRANSACTION_FEE)
        values.append(state.portfolio_value)

    actions_arr = np.array(actions)
    equity = np.array(values)
    returns = np.diff(equity) / equity[:-1]
    returns = np.insert(returns, 0, 0.0)

    total_return = ((equity[-1] - equity[0]) / equity[0]) * 100.0
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    
    pct_short = (np.sum(actions_arr == 0) / len(actions_arr)) * 100.0
    pct_neutral = (np.sum(actions_arr == 1) / len(actions_arr)) * 100.0
    pct_long = (np.sum(actions_arr == 2) / len(actions_arr)) * 100.0

    return {
        "Return (%)": total_return,
        "Sharpe": sharpe,
        "% Short": pct_short,
        "% Neutral": pct_neutral,
        "% Long": pct_long,
    }


if __name__ == "__main__":
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    prices, processed_df = extract_price_series(test_df)

    ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    if not ckpt_files:
        print(f"No checkpoint files found in {CHECKPOINT_DIR}")
        exit()

    print(f"Found {len(ckpt_files)} checkpoints to evaluate.\n")

    results = {}
    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path)
        try:
            model = torch.load(ckpt_path, map_location="cpu")
            metrics = evaluate_single_checkpoint(processed_df, prices, model)
            results[ckpt_name] = metrics
        except Exception as e:
            print(f"Skipping {ckpt_name} due to error: {e}")

    summary_df = pd.DataFrame(results).T
    print("\n================ CHECKPOINT EVOLUTION SUMMARY ================")
    print(summary_df.to_string(formatters={
        "Return (%)": "{:+.2f}%".format,
        "Sharpe": "{:.2f}".format,
        "% Short": "{:.1f}%".format,
        "% Neutral": "{:.1f}%".format,
        "% Long": "{:.1f}%".format,
    }))
    print("==============================================================\n")