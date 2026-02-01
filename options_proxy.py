def protective_put_proxy(weekly_ret, premium_bps=20, crash_cap=0.03):
    premium = premium_bps/10000.0
    r = weekly_ret - premium
    if r < -crash_cap:
        r = -crash_cap - premium
    return r
