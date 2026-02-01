import numpy as np

def sector_scores(sectors: dict[str, list[str]], returns_5d: dict[str, float]) -> dict[str, float]:
    """
    Score sector using top-constituent momentum and dispersion penalty.
    """
    out = {}
    for s, tks in sectors.items():
        rets = [returns_5d.get(t, np.nan) for t in tks]
        rets = [r for r in rets if r == r]
        if not rets:
            out[s] = -1e9
            continue
        rets_sorted = sorted(rets, reverse=True)
        top = rets_sorted[: max(1, min(3, len(rets_sorted)))]
        score = float(np.mean(top) - 0.3 * np.std(rets))
        out[s] = score
    return out
