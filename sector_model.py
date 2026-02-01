import numpy as np

def sector_scores(sectors: dict[str, list[str]], prices: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    prices[sector][ticker] = recent_return (e.g., last 5d)
    score = mean(top returns) - penalty for dispersion (risk)
    """
    out = {}
    for s, tks in sectors.items():
        rets = [prices.get(s, {}).get(t, np.nan) for t in tks]
        rets = [r for r in rets if r == r]
        if not rets:
            out[s] = -1e9
            continue
        rets_sorted = sorted(rets, reverse=True)
        top = rets_sorted[: max(1, min(3, len(rets_sorted)))]
        score = float(np.mean(top) - 0.3 * np.std(rets))
        out[s] = score
    return out
