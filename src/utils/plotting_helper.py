import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_drawdown(equity_series):
    """Calculate drawdown percentage over time."""
    running_max = equity_series.cummax()
    return (equity_series - running_max) / running_max


def compute_metrics(equity_series):
    """Calculate Total Return, Sharpe Ratio, and Max Drawdown."""
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    daily_returns = equity_series.pct_change().dropna()

    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() != 0
        else 0
    )
    max_dd = compute_drawdown(equity_series).min()

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
    }


def load_and_plot_real_logs(price_file, logs_dir):
    """Loads actual dataset prices and evaluation log CSVs from logs/evaluation/."""

    # 1. Load Close Price Data from processed or raw CSV
    if not os.path.exists(price_file):
        raise FileNotFoundError(f"Price file not found at: {price_file}")

    price_df = pd.read_csv(price_file)
    price_col = next(
        (
            col
            for col in ["close", "Close", "adj_close", "Adj Close"]
            if col in price_df.columns
        ),
        None,
    )

    if price_col is None:
        # Fallback to the first numeric column if named differently
        price_col = price_df.select_dtypes(include=[np.number]).columns[0]

    df = pd.DataFrame()
    df["Price"] = price_df[price_col].values

    # 2. Strategy 1: Buy and Hold
    df["Buy_Hold"] = df["Price"] / df["Price"].iloc[0]

    # 3. Strategy 2: Simple Moving Average (SMA Crossover)
    sma_window = 20
    df["SMA"] = df["Price"].rolling(window=sma_window).mean()
    df["Signal"] = np.where(df["Price"] > df["SMA"], 1.0, 0.0)
    df["SMA_Return"] = (
        pd.Series(df["Price"]).pct_change() * df["Signal"].shift(1)
    )
    df["SMA_Strategy"] = (1 + df["SMA_Return"].fillna(0)).cumprod()

    # 4. Load Real Agent Log CSVs from logs/evaluation/
    log_mappings = {
        "PPO": os.path.join(logs_dir, "test_ppo_equity.csv"),
        "MuZero": os.path.join(logs_dir, "test_muzero_equity.csv"),
    }

    strategies = ["Buy_Hold", "SMA_Strategy"]

    for model_name, filepath in log_mappings.items():
        if os.path.exists(filepath):
            model_df = pd.read_csv(filepath)
            # Find equity/portfolio value column dynamically
            val_col = next(
                (
                    col
                    for col in ["portfolio_value", "equity", "total_value", "value"]
                    if col in model_df.columns
                ),
                model_df.columns[0],
            )
            df[model_name] = model_df[val_col].values[: len(df)]
            # Normalize starting value to 1.0
            df[model_name] = df[model_name] / df[model_name].iloc[0]
            strategies.append(model_name)
        else:
            print(f"Warning: Log file for {model_name} not found at {filepath}")

    # 5. Calculate Metrics
    metrics = {s: compute_metrics(df[s]) for s in strategies}
    metrics_df = pd.DataFrame(metrics).T
    print("\n--- Performance Summary (Real Evaluation Logs) ---")
    print(metrics_df)

    # 6. Plot Results
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for s in strategies:
        axes[0].plot(df.index, df[s], label=s, linewidth=2)

    axes[0].set_ylabel("Normalized Portfolio Value")
    axes[0].set_title("Real Evaluation Log Comparison")
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    for s in strategies:
        dd = compute_drawdown(df[s])
        axes[1].plot(df.index, dd * 100, label=s, linewidth=1.5)

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Time Step")
    axes[1].set_title("Underwater Chart (Risk & Drawdown)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Point directly to the actual files identified in your tree
    PRICE_DATA_PATH = "data/processed/test_data.csv"
    EVAL_LOGS_DIR = "logs/evaluation"

    load_and_plot_real_logs(PRICE_DATA_PATH, EVAL_LOGS_DIR)