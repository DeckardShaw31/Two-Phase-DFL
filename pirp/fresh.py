"""Demand episodes from FreshRetailNet-50K (Wang et al. 2025, CC BY 4.0) for the real-data case.

The dataset has 50,000 store-product series of a fresh-food retailer (97 days, hourly sales
and hourly out-of-stock status, discount, holiday flag and weather). An episode is one product
sold at ten stores of the same city over the last 60 days (35 history + 21 decision + 4 tail).

* Selection: series without missing days and with fewer than 10% zero-sales days; at most
  one ten-store group per (city, product); products, not groups, are assigned to the
  train/validation/test splits so that no product appears in two splits. Fleet and shelf
  capacities are dimensioned on the network's average volume, so groups whose sales level
  shifts by more than -20%/+25% between the history and decision windows (e.g. products
  entering their season) are excluded: in them capacity, not the replenishment policy,
  would bind. This choice concerns only which data enter the experiment; forecasts and
  policies use past information only.
* Censoring: on a day with out-of-stock hours, sales understate demand. Latent demand is
  sales divided by the share of the series' typical intraday sales that falls in in-stock
  hours (profile estimated on the series' fully in-stock days), with the share floored at
  0.25 so that a day can at most be scaled up fourfold.
* Scale: the network, fleet and shelf capacities are those of the synthetic study, so each
  store's latent demand is rescaled to the store's base volume using the mean of the
  history window only (no information from the decision days).
* Covariates: discount (1 - price ratio), holiday flag, weekday and the city's average
  temperature. Observed temperature is used as its own forecast.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

WINDOW = 60
LIGHT = ["city_id", "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount",
         "holiday_flag", "avg_temperature"]


def _read(raw_dir: str, columns: list[str], products=None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    filt = [("product_id", "in", sorted(int(p) for p in products))] if products is not None else None
    parts = [pq.read_table(os.path.join(raw_dir, f), columns=columns, filters=filt).to_pandas()
             for f in ("train.parquet", "eval.parquet")]
    return pd.concat(parts, ignore_index=True)


def select_groups(raw_dir: str, n_split: dict, hist: int, n_dec: int, seed: int = 0,
                  level_band: tuple = (0.8, 1.25)) -> list[dict]:
    df = _read(raw_dir, LIGHT)
    dates = np.sort(df.dt.unique())[-WINDOW:]
    period = pd.Series(np.where(np.isin(dates, dates[:hist]), "h", np.where(np.isin(dates, dates[hist:hist + n_dec]),
                                                                          "d", "x")), index=dates)
    win = df[df.dt.isin(dates)].assign(period=lambda x: x.dt.map(period))
    lvl = win[win.period != "x"].groupby(["city_id", "product_id", "store_id", "period"]).sale_amount.sum().unstack()
    g = df.groupby(["city_id", "product_id", "store_id"]).agg(
        n=("dt", "size"), zero=("sale_amount", lambda s: float((s == 0).mean()))).reset_index()
    ok = g[(g.n == df.dt.nunique()) & (g.zero < 0.10)]
    rng = np.random.default_rng(seed)
    groups = []
    for (city, prod), sub in ok.groupby(["city_id", "product_id"]):
        if len(sub) >= 10:
            stores = np.sort(rng.choice(sub.store_id.values, 10, replace=False))
            lv = lvl.loc[[(city, prod, st) for st in stores]].sum()
            ratio = (lv["d"] / n_dec) / max(lv["h"] / hist, 1e-9)
            if not level_band[0] <= ratio <= level_band[1]:
                continue
            groups.append({"city_id": int(city), "product_id": int(prod), "stores": stores.tolist()})
    products = np.array(sorted({gr["product_id"] for gr in groups}))
    rng.shuffle(products)
    by_prod = {p: [gr for gr in groups if gr["product_id"] == p] for p in products}
    out, it = [], iter(products)
    for split in ("test", "val", "train"):
        chosen = []
        while len(chosen) < n_split[split]:
            p = next(it)
            for gr in by_prod[p]:
                if len(chosen) < n_split[split]:
                    chosen.append(dict(gr, split=split))
        out += chosen
    return out


def _latent(sales: np.ndarray, hours: np.ndarray, status: np.ndarray) -> np.ndarray:
    """sales (T,), hours (T,24) hourly sales, status (T,24) 1 = out of stock."""
    full = status.sum(1) == 0
    tot = hours[full].sum()
    prof = hours[full].sum(0) / tot if full.any() and tot > 0 else np.full(24, 1 / 24)
    share = (prof[None, :] * (1 - status)).sum(1)
    return sales / np.maximum(share, 0.25)


def build_episodes(raw_dir: str, bases: np.ndarray, n_split: dict, hist: int, n_dec: int = 21, seed: int = 0) -> dict:
    groups = select_groups(raw_dir, n_split, hist, n_dec, seed)
    prods = {gr["product_id"] for gr in groups}
    df = _read(raw_dir, LIGHT + ["hours_sale", "hours_stock_status"], products=prods)
    df = df.sort_values("dt")
    data = {"train": [], "val": [], "test": []}
    for gr in groups:
        sel = df[(df.product_id == gr["product_id"]) & df.store_id.isin(gr["stores"]) & (df.city_id == gr["city_id"])]
        dem, disc, censored = [], [], []
        for s in gr["stores"]:
            r = sel[sel.store_id == s]  # full series (profile estimated on all days); window cut below
            lat = _latent(r.sale_amount.values, np.stack(r.hours_sale.values), np.stack(r.hours_stock_status.values))
            dem.append(lat[-WINDOW:])
            disc.append(np.clip(1.0 - r.discount.values[-WINDOW:], 0.0, 0.9))
            censored.append((r.stock_hour6_22_cnt.values[-WINDOW:] > 0).mean())
        dem = np.array(dem)
        dem = dem * (bases / np.maximum(dem[:, :hist].mean(1), 1e-9))[:, None]
        days = sel[sel.store_id == gr["stores"][0]].dt.values[-WINDOW:]
        daily = sel.groupby("dt").agg(temp=("avg_temperature", "mean"), hol=("holiday_flag", "max")).loc[days]
        data[gr["split"]].append({
            "demand": dem, "mu": dem.copy(), "dow": pd.to_datetime(days).dayofweek.values,
            "temp": daily.temp.values.astype(float), "holiday": daily.hol.values.astype(float),
            "disc": np.array(disc), "temp_fc_err": np.zeros((WINDOW, 4)),
            "meta": {"city_id": gr["city_id"], "product_id": gr["product_id"], "stores": gr["stores"],
                     "censored_day_share": float(np.mean(censored))}})
    return data
