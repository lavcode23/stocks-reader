def apply_protective_put_proxy(weekly_return: float, premium_bps: float, crash_protect_pct: float) -> float:
    premium = premium_bps / 10000.0
    r = weekly_return - premium
    # if crash worse than threshold, cap the loss
    if r < -crash_protect_pct:
        r = -crash_protect_pct - premium
    return r
