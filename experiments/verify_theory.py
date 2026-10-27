"""Numerical verification of the analytical results (Section 3 of the revised manuscript).

T1  Complete recourse: every lookahead MILP is feasible for arbitrary non-negative demand.
T2  The recourse LP with an early-disposal option has an optimal solution with FIFO issuing and
    greedy service (checked by solving the recourse with FIFO and greedy service imposed through
    binary indicators and comparing values) and its value never exceeds the physical FIFO cost.
P3  Marginal value of a delivery: LP duals of the fresh-stock rows lie in [-p_u, p_s] and are
    subgradients of V_LP; for h <= p_s the FIFO cost's derivative with respect to the delivery
    of day t lies in [-p_u, p_s + (H-1-t)h].
P4  DFL fixed point: with one-day shelf life the decision-focused gradient vanishes at the
    perishable newsvendor quantile p_u / (p_u + p_s).
P5  Coverage layer: deliveries implied by the MILP visit pattern (``coverage_quantities``)
    reproduce the MILP's planned quantities.

Run:  python -m experiments.verify_theory --out results/theory
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from pirp.dfl import coverage_quantities
from pirp.milp import PIRPModel, _Builder
from pirp.network import build_network, held_karp_all_subsets
from pirp.simulator import CostParams, simulate_plan, torch_fifo_cost


def recourse_lp(B0, q, d, m, costs, disposal=True, fifo_eps=0.0, return_x=False):
    """Single-store recourse LP with optional early disposal. Returns (value, duals of fresh rows).

    fifo_eps > 0 adds a vanishing reward for selling older cohorts first (tie-breaking only).
    """
    H = len(q)
    b = _Builder()
    h, ps, pu = costs.h, costs.p_waste, costs.p_lost
    s = np.zeros((H, m), int); w = np.zeros((H, m), int); z = np.zeros((H, m), int); u = np.zeros(H, int)
    fresh = []
    for k in range(H):
        cs = np.array([ps] + [h] * (m - 1))
        s[k] = b.var(m, 0, np.inf, cs)
        w[k] = b.var(m, 0, np.inf, -cs - fifo_eps * (m - np.arange(m)))
        z[k] = b.var(m, 0, np.inf if disposal else 0.0, np.array([0.0] + [ps - h] * (m - 1)))
        u[k] = b.var(1, 0, np.inf, pu)[0]
    for k in range(H):
        b.row(list(w[k]) + [u[k]], [1] * (m + 1), d[k], d[k])
        for r in range(m):
            b.row([w[k][r], s[k][r], z[k][r]], [1, -1, 1], -np.inf, 0)
        fresh.append(b.row([s[k][m - 1]], [1], q[k], q[k]))
        if k == 0:
            for r in range(m - 1):
                b.row([s[0][r]], [1], B0[r], B0[r])
        else:
            for r in range(m - 1):
                b.row([s[k][r], s[k - 1][r + 1], w[k - 1][r + 1], z[k - 1][r + 1]], [1, -1, 1, 1], 0, 0)
    A = coo_matrix((b.vals, (b.rows, b.cols)), shape=(b.nr, b.nv)).tocsr()
    bl, bu = np.array(b.bl), np.array(b.bu)
    eq = bl == bu
    res = linprog(np.array(b.c), A_ub=A[~eq], b_ub=bu[~eq], A_eq=A[eq], b_eq=bl[eq],
                  bounds=list(zip(b.lb, b.ub)), method="highs")
    pos = np.cumsum(eq) - 1
    pi = np.array([res.eqlin.marginals[pos[r]] for r in fresh])
    if return_x:
        return float(res.fun), pi, res.x, (s, w, z)
    return float(res.fun), pi


def fifo_complementarity_violation(x, idx, m, H):
    """Max over days of min(older stock left unsold & undisposed, younger stock sold)."""
    s, w, z = idx
    viol = 0.0
    for k in range(H):
        left = x[s[k]] - x[w[k]] - x[z[k]]
        sold = x[w[k]]
        for r_old in range(m):
            for r_young in range(r_old + 1, m):
                viol = max(viol, min(left[r_old], sold[r_young]))
    return viol


def fifo_constrained_value(B0, q, d, m, costs):
    """Recourse value when sales must follow FIFO and demand must be served while stock is left.

    For each day k and cohort r a binary b[k,r] = 1 allows sales from cohort r; it requires every
    older cohort r' < r to be exhausted (sold or disposed). A binary g[k] = 1 allows lost sales and
    requires all stock to be exhausted.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    H = len(q)
    b = _Builder()
    h, ps, pu = costs.h, costs.p_waste, costs.p_lost
    M = float(B0.sum() + q.sum() + d.sum() + 1.0)
    s = np.zeros((H, m), int); w = np.zeros((H, m), int); z = np.zeros((H, m), int); u = np.zeros(H, int)
    bb = np.zeros((H, m), int); gg = np.zeros(H, int)
    for k in range(H):
        cs = np.array([ps] + [h] * (m - 1))
        s[k] = b.var(m, 0, np.inf, cs)
        w[k] = b.var(m, 0, np.inf, -cs)
        z[k] = b.var(m, 0, np.inf, np.array([0.0] + [ps - h] * (m - 1)))
        u[k] = b.var(1, 0, np.inf, pu)[0]
        bb[k] = b.var(m, 0, 1, 0.0, integer=True)
        gg[k] = b.var(1, 0, 1, 0.0, integer=True)[0]
    for k in range(H):
        b.row(list(w[k]) + [u[k]], [1] * (m + 1), d[k], d[k])
        for r in range(m):
            b.row([w[k][r], s[k][r], z[k][r]], [1, -1, 1], -np.inf, 0)
            b.row([w[k][r], bb[k][r]], [1, -M], -np.inf, 0)
            for r2 in range(r):
                b.row([s[k][r2], w[k][r2], z[k][r2], bb[k][r]], [1, -1, -1, M], -np.inf, M)
        b.row([u[k], gg[k]], [1, -M], -np.inf, 0)
        for r in range(m):
            b.row([s[k][r], w[k][r], z[k][r], gg[k]], [1, -1, -1, M], -np.inf, M)
        b.row([s[k][m - 1]], [1], q[k], q[k])
        if k == 0:
            for r in range(m - 1):
                b.row([s[0][r]], [1], B0[r], B0[r])
        else:
            for r in range(m - 1):
                b.row([s[k][r], s[k - 1][r + 1], w[k - 1][r + 1], z[k - 1][r + 1]], [1, -1, 1, 1], 0, 0)
    A = coo_matrix((b.vals, (b.rows, b.cols)), shape=(b.nr, b.nv)).tocsr()
    res = milp(np.array(b.c), integrality=np.array(b.integ), bounds=Bounds(np.array(b.lb), np.array(b.ub)),
               constraints=LinearConstraint(A, np.array(b.bl), np.array(b.bu)), options={"mip_rel_gap": 1e-9})
    return float(res.fun)


def fifo_cost_and_omega(B0, q, d, m, costs):
    out = simulate_plan(B0[None, :], q[None, :], d[None, :], costs)
    waste = out["waste"][0]
    omega = float(sum(waste[t] * min(m - 1, t) for t in range(len(q))))
    return out["inv_cost"], omega


def check_T2_P3(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(n):
        m = int(rng.integers(2, 5)); H = int(rng.integers(2, 6))
        costs = CostParams(h=float(rng.choice([0.5, 1.0, 2.0])), p_waste=float(rng.choice([2.0, 5.0, 10.0])),
                           p_lost=float(rng.choice([10.0, 20.0, 40.0])))
        B0 = rng.integers(0, 12, m - 1).astype(float)
        q = rng.integers(0, 30, H).astype(float) * (rng.random(H) < 0.7)
        d = rng.poisson(rng.uniform(3, 20), H).astype(float)
        v_lp, pi = recourse_lp(B0, q, d, m, costs, disposal=True)
        v_fifo = fifo_constrained_value(B0, q, d, m, costs)
        fifo_viol = abs(v_fifo - v_lp)
        c_fifo, omega = fifo_cost_and_omega(B0, q, d, m, costs)
        Qt = torch.tensor(q[None, None, :], requires_grad=True)
        ct = torch_fifo_cost(torch.tensor(B0[None, None, :]), Qt, torch.tensor(d[None, None, :]), costs)
        ct.sum().backward()
        g = Qt.grad.numpy().ravel()
        # subgradient check of V_LP at random directions
        sub_ok = True
        for _ in range(3):
            dq = rng.normal(0, 3, H)
            q2 = np.maximum(q + dq, 0)
            v2, _ = recourse_lp(B0, q2, d, m, costs, disposal=True)
            if v2 < v_lp + pi @ (q2 - q) - 1e-6:
                sub_ok = False
        bound_hi = costs.p_waste + (H - 1 - np.arange(H)) * costs.h
        rows.append({"m": m, "H": H, "v_lp": v_lp, "c_fifo": c_fifo, "omega": omega,
                     "fifo_violation": fifo_viol, "v_fifo_constrained": v_fifo,
                     "lp_le_fifo": v_lp <= c_fifo + 1e-7, "rel_gap": (c_fifo - v_lp) / max(c_fifo, 1e-9),
                     "pi_min": float(pi.min()), "pi_max": float(pi.max()), "pi_in_bounds":
                         bool((pi >= -costs.p_lost - 1e-7).all() and (pi <= costs.p_waste + 1e-7).all()),
                     "grad_in_bounds": bool((g >= -costs.p_lost - 1e-7).all() and (g <= bound_hi + 1e-7).all()),
                     "grad_bound_tight": bool(np.isclose(g, bound_hi).any() or np.isclose(g, -costs.p_lost).any()),
                     "subgradient_ok": sub_ok})
    return rows


def check_T1(n=60, seed=1):
    rng = np.random.default_rng(seed)
    _, dist = build_network("R1")
    tc = held_karp_all_subsets(dist)
    statuses = []
    for m, H, S in ((2, 2, 1), (3, 3, 1), (4, 3, 1), (3, 3, 5)):
        mdl = PIRPModel(dist, np.full(10, 50.0), 2, 120.0, m, H, CostParams(), n_scen=S, tour_cost=tc)
        for _ in range(n // 4):
            B0 = rng.integers(0, 25, (10, m - 1)).astype(float)
            B0 = np.minimum(B0, 50.0 / max(m - 1, 1))
            # extreme demands, including zero and far above fleet capacity
            scale = rng.choice([0.0, 5.0, 20.0, 80.0])
            d = rng.gamma(2, scale / 2 + 1e-9, (S, 10, H)) if S > 1 else rng.gamma(2, scale / 2 + 1e-9, (10, H))
            r = mdl.solve(B0, d, time_limit=30, mip_rel_gap=0.01)
            statuses.append({"m": m, "H": H, "S": S, "status": int(r.status), "feasible": bool(np.isfinite(r.objective)),
                             "gap": r.mip_gap, "secs": r.seconds})
    return statuses


def check_P4(seed=2, n=20000, iters=3000):
    """One-day shelf life, single store: DFL gradient descent converges to F(q) = p_u/(p_u+p_s)."""
    rng = np.random.default_rng(seed)
    costs = CostParams(h=1.0, p_waste=5.0, p_lost=20.0)
    d = rng.negative_binomial(8, 8 / (8 + 15.0), n).astype(np.float32)
    target = torch.tensor(15.0, requires_grad=True)
    opt = torch.optim.SGD([target], lr=0.05)
    D = torch.tensor(d)
    for it in range(iters):
        idx = torch.as_tensor(rng.integers(0, n, 256))
        q = target.expand(len(idx))
        cost = torch_fifo_cost(torch.zeros(len(idx), 1, 0), q[:, None, None], D[idx][:, None, None], costs).mean()
        opt.zero_grad(); cost.backward(); opt.step()
    crit = costs.p_lost / (costs.p_lost + costs.p_waste)
    return {"learned_target": float(target.detach()), "empirical_quantile_at_critical": float(np.quantile(d, crit)),
            "critical_ratio": crit, "cdf_at_learned": float(np.mean(d <= float(target)))}


def check_P5(n=40, seed=3):
    rng = np.random.default_rng(seed)
    _, dist = build_network("R1")
    tc = held_karp_all_subsets(dist)
    costs = CostParams()
    cap = np.full(10, 55.0)
    mdl = PIRPModel(dist, cap, 2, 120.0, 3, 3, costs, tour_cost=tc)
    diffs, rel = [], []
    for _ in range(n):
        B0 = rng.integers(0, 10, (10, 2)).astype(float)
        d = rng.gamma(6, 3, (10, 3))
        r = mdl.solve(B0, d, time_limit=30, mip_rel_gap=0.001)
        veh = -np.ones((10, 3), int)
        for k, masks in enumerate(r.info["routes"]):
            for v, mk in enumerate(masks):
                for i in range(10):
                    if (mk >> i) & 1:
                        veh[i, k] = v
        qrule = coverage_quantities(torch.tensor(d), (veh >= 0).astype(float), veh, torch.tensor(B0),
                                    torch.tensor(cap), 120.0, 3).numpy()
        diffs.append(np.abs(qrule - r.q).max())
        rel.append(np.abs(qrule - r.q).sum() / max(r.q.sum(), 1e-9))
    return {"max_abs_diff": float(np.max(diffs)), "median_max_abs_diff": float(np.median(diffs)),
            "mean_rel_l1": float(np.mean(rel)), "share_exact(<0.5 unit)": float(np.mean(np.array(diffs) < 0.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/theory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    t2 = pd.DataFrame(check_T2_P3())
    t2.to_csv(os.path.join(a.out, "t2_p3_recourse_checks.csv"), index=False)
    t1 = pd.DataFrame(check_T1())
    t1.to_csv(os.path.join(a.out, "t1_complete_recourse.csv"), index=False)
    p4 = check_P4()
    p5 = check_P5()
    summary = {
        "T1_instances": int(len(t1)), "T1_all_feasible": bool(t1.feasible.all()),
        "T1_status_counts": t1.status.value_counts().to_dict(),
        "T2_instances": int(len(t2)), "T2_max_abs_diff_fifo_constrained_vs_lp": float(t2.fifo_violation.max()),
        "T2_lp_le_fifo_all": bool(t2.lp_le_fifo.all()), "T2_share_equal": float((t2.c_fifo - t2.v_lp < 1e-6).mean()),
        "T2_median_rel_gap": float(t2.rel_gap.median()), "T2_mean_rel_gap": float(t2.rel_gap.mean()),
        "P3_dual_bounds_all": bool(t2.pi_in_bounds.all()), "P3_subgradient_all": bool(t2.subgradient_ok.all()),
        "P3_fifo_grad_bounds_all": bool(t2.grad_in_bounds.all()), "P4": p4, "P5": p5,
    }
    with open(os.path.join(a.out, "theory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
