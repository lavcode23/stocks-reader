from dataclasses import dataclass
import pandas as pd


@dataclass
class TradePlan:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    take_price: float
    planned_exit_date: pd.Timestamp


@dataclass
class TradeResult:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    gross_return: float


def make_trade_plan(
    ticker,
    prices,
    signal_date,
    holding_days,
    stop_loss_pct,
    take_profit_pct,
):
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)

    future = prices.loc[prices.index > signal_date]
    if future.empty:
        return None

    entry_date = future.index[0]
    entry_price = float(future.loc[entry_date, "Open"])

    stop_price = entry_price * (1 - stop_loss_pct)
    take_price = entry_price * (1 + take_profit_pct)

    after = prices.loc[prices.index >= entry_date]
    planned_exit = after.index[min(len(after) - 1, holding_days)]

    return TradePlan(
        ticker,
        entry_date,
        entry_price,
        stop_price,
        take_price,
        planned_exit,
    )


def simulate_trade(prices, plan):
    df = prices.copy()
    df.index = pd.to_datetime(df.index)

    window = df.loc[(df.index >= plan.entry_date) & (df.index <= plan.planned_exit_date)]

    exit_date = window.index[-1]
    exit_price = float(window.iloc[-1]["Close"])
    reason = "time_exit"

    for d, row in window.iterrows():
        if row["Low"] <= plan.stop_price:
            exit_date = d
            exit_price = plan.stop_price
            reason = "stop_loss"
            break
        if row["High"] >= plan.take_price:
            exit_date = d
            exit_price = plan.take_price
            reason = "take_profit"
            break

    ret = exit_price / plan.entry_price - 1

    return TradeResult(
        plan.ticker,
        plan.entry_date,
        exit_date,
        plan.entry_price,
        exit_price,
        reason,
        float(ret),
    )
