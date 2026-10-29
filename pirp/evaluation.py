"""Closed-loop evaluation, parallel execution and paired statistics."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from .network import SubsetRouting
from .policies import BaseStockPolicy, Context, Instance, LookaheadPolicy, ULPolicy
from .simulator import step_fifo

_CACHE: dict = {}


def make_policy(spec: dict, inst: Instance, router: SubsetRouting):
    kind = spec["kind"]
    if kind == "bsp":
        return BaseStockPolicy(inst, router, R=spec["R"], alpha=spec["alpha"], waste_aware=False)
    if kind == "bsp_ew":
        return BaseStockPolicy(inst, router, R=spec["R"], alpha=spec["alpha"], waste_aware=True)
    if kind == "ul":
        return ULPolicy(inst, router, lam=spec["lam"], alpha=spec["alpha"], beta=spec["beta"])
    if kind in ("ul_dfl", "nn_dfl"):  # the model's own kind decides rule correction vs direct order
        from .rule_dfl import ULDFLModel, ULDFLPolicy
        return ULDFLPolicy(inst, router, ULDFLModel.from_state(spec["dfl_state"]))
    if kind == "lookahead":
        dfl = None
        if spec["target"] == "dfl":
            from .dfl import DFLTargetModel
            dfl = DFLTargetModel.from_state(spec["dfl_state"])
        return LookaheadPolicy(inst, router, spec["target"], param=spec.get("param", 0.0),
                               n_scen=spec.get("n_scen", 1), time_limit=spec.get("time_limit", 20.0),
                               gap=spec.get("gap", 0.005), dfl_model=dfl,
                               integer_days=spec.get("integer_days"), sampler=spec.get("sampler", "bands"))
    raise ValueError(kind)


def run_episode(policy, inst: Instance, router: SubsetRouting, demand: np.ndarray, fc: dict, start: int,
                n_days: int, seed: int = 0, record_states: bool = False) -> dict:
    """Run one closed-loop episode. demand: (N, T_total); fc: per-planning-day forecasts."""
    N, m = inst.N, inst.shelf_life
    H = inst.horizon
    B = np.zeros((N, m - 1))
    rec = {k: [] for k in ("q", "route_cost", "lost", "waste", "carried", "sold", "demand", "n_routes",
                           "status", "gap", "secs", "fallback", "visits")}
    states = []
    for day in range(n_days):
        t = start + day
        ctx = Context(day=day, B=B.copy(), mu=fc["mu"][day], qs=fc["qs"][day], X=fc["X"][day],
                      future=demand[:, t:t + H], seed=seed * 1000 + day)
        q, info = policy.decide(ctx)
        q = np.maximum(q, 0.0)
        q[q < 1e-6] = 0.0
        packed = router.pack(q)
        if packed is None:  # should not happen; enforce feasibility explicitly
            from .policies import repair
            q = repair(q, q, router, inst, B)
            packed = router.pack(q)
        rc, masks = packed
        if record_states:
            states.append({"day": day, "B": B.copy(), "q": q.copy()})
        B, sold, lost, waste, carried = step_fifo(B, q, demand[:, t], inst.lifo_share)
        rec["q"].append(q)
        rec["route_cost"].append(inst.costs.c_route * rc)
        rec["n_routes"].append(len(masks))
        rec["lost"].append(lost)
        rec["waste"].append(waste)
        rec["carried"].append(carried)
        rec["sold"].append(sold)
        rec["demand"].append(demand[:, t])
        rec["visits"].append((q > 0).astype(float))
        rec["status"].append(info.get("status", -1))
        rec["gap"].append(info.get("gap", np.nan))
        rec["secs"].append(info.get("secs", 0.0))
        rec["fallback"].append(bool(info.get("fallback", False)))
    out = {k: np.array(v) for k, v in rec.items()}
    c = inst.costs
    out["cost_route"] = float(out["route_cost"].sum())
    out["cost_hold"] = float(c.h * out["carried"].sum())
    out["cost_waste"] = float(c.p_waste * out["waste"].sum())
    out["cost_lost"] = float(c.p_lost * out["lost"].sum())
    out["cost_total"] = out["cost_route"] + out["cost_hold"] + out["cost_waste"] + out["cost_lost"]
    out["fill_rate"] = float(out["sold"].sum() / max(out["demand"].sum(), 1e-9))
    out["waste_rate"] = float(out["waste"].sum() / max(out["q"].sum(), 1e-9))
    if record_states:
        out["states"] = states
    return out


def _worker(task):
    spec, inst, demand, fc, start, n_days, seed, key = task
    if "router" not in _CACHE or _CACHE.get("router_key") != key:
        _CACHE["router"] = SubsetRouting(inst.dist, inst.n_vehicles, inst.vehicle_cap)
        _CACHE["router_key"] = key
        _CACHE["policies"] = {}
    router = _CACHE["router"]
    pkey = repr(sorted((k, v) for k, v in spec.items() if k != "dfl_state")) + str(id(spec.get("dfl_state")))
    pol = _CACHE["policies"].get(pkey)
    if pol is None:
        pol = make_policy(spec, inst, router)
        _CACHE["policies"][pkey] = pol
    res = run_episode(pol, inst, router, demand, fc, start, n_days, seed)
    keep = ("cost_total", "cost_route", "cost_hold", "cost_waste", "cost_lost", "fill_rate", "waste_rate")
    summary = {k: res[k] for k in keep}
    summary.update({
        "lost_units": float(res["lost"].sum()), "waste_units": float(res["waste"].sum()),
        "delivered": float(res["q"].sum()), "carried_mean": float(res["carried"].mean()),
        "routes": float(res["n_routes"].sum()), "visits": float(res["visits"].sum()),
        "solver_secs": float(res["secs"].sum()), "max_gap": float(np.nanmax(res["gap"])) if np.isfinite(res["gap"]).any() else np.nan,
        "n_fallback": int(res["fallback"].sum()), "n_nonoptimal": int(np.sum((res["status"] != 0) & (res["status"] != -1))),
        "store_fill": (res["sold"].sum(0) / np.maximum(res["demand"].sum(0), 1e-9)).tolist(),
        "store_waste": res["waste"].sum(0).tolist(), "store_lost": res["lost"].sum(0).tolist(),
    })
    return summary


_POOL: dict = {}


def get_pool(n_workers: int) -> ProcessPoolExecutor:
    """One persistent worker pool per process (creating and destroying many pools in quick
    succession triggers a known handle race of ProcessPoolExecutor on Windows)."""
    pool = _POOL.get(n_workers)
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=n_workers)
        _POOL[n_workers] = pool
    return pool


def shutdown_pools() -> None:
    for pool in _POOL.values():
        pool.shutdown(wait=True)
    _POOL.clear()


def run_many(tasks: list, n_workers: int | None = None) -> list:
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        return [_worker(t) for t in tasks]
    return list(get_pool(n_workers).map(_worker, tasks, chunksize=1))


# --------------------------------------------------------------------------- statistics
def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def paired_test(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Paired comparison of per-episode costs a - b (negative favours a)."""
    d = np.asarray(a) - np.asarray(b)
    lo, hi = paired_bootstrap(d, seed=seed)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(d)
        pw = float(w.pvalue)
    except ValueError:
        pw = 1.0
    rel = d / np.asarray(b)
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "ci_lo": lo, "ci_hi": hi,
            "p_t": float(t.pvalue), "p_wilcoxon": pw, "mean_rel_diff": float(rel.mean()),
            "win_rate": float(np.mean(d < 0))}
