import numpy as np
import pandas as pd


def make_bid(log):
    return pd.Series([foo[1]['bids']['best'] for foo in log], index=[foo[0] for foo in log])


def make_ask(log):
    return pd.Series([foo[1]['asks']['best'] for foo in log], index=[foo[0] for foo in log])


def make_mid(log):
    return 0.5 * (make_bid(log) + make_ask(log))


def future_value(mid, horizon):
    res = mid.reindex(mid.index + horizon, method='pad')
    res.index = mid.index
    return res


def future_return(mid, horizon):
    return mid.reindex(mid.index + horizon, method='pad').values - mid


def conservative_future_return(bid, ask, horizon):
    pos_part = future_value(bid, horizon) - ask
    pos_part = pos_part.where(pos_part > 0, other=0.0)
    neg_part = future_value(ask, horizon) - bid
    neg_part = neg_part.where(neg_part < 0, other=0.0)
    return pos_part + neg_part


def book_price(qty, bids, asks):
    def impl(lob, sign):
        tot_price = 0
        tot_size = 0
        level = 0 if sign > 0 else len(lob) - 1
        while tot_size < qty and level >= 0 and level < len(lob):
            lv_price = lob[level][0]
            lv_size = lob[level][1]
            eff_size = lv_size if lv_size < qty - tot_size else qty - tot_size
            tot_price += eff_size * lv_price
            tot_size += eff_size
            level += sign
        if tot_size == qty:
            return tot_price / tot_size
        else:
            return None
    bp = impl(bids, -1)
    ap = impl(asks, 1)
    if bp is None or ap is None:
        return None
    else:
        return 0.5 * (bp + ap)


def ema(series, dur, reset_on_nan=False):
    def gen():
        last = np.nan
        last_t = np.nan
        last_v = np.nan
        for t, v in zip(series.index, series):
            # Decay towards last_v
            if np.isfinite(last_v):
                # last must also be finite
                k = np.exp((last_t - t) / dur)
                last = last * k + last_v * (1 - k)
            # Process NaNs
            if not np.isfinite(v) and reset_on_nan:
                last = v
            if not np.isfinite(last) and np.isfinite(v):
                last = v
            # Remember state
            last_v = v
            last_t = t
            yield last
    return pd.Series(gen(), index=series.index)
