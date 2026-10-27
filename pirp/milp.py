"""Rolling-horizon perishable inventory-routing MILPs solved with HiGHS (SciPy).

Two lookahead models share one sparse template builder:

* ``deterministic``: one planning demand path (point, quantile, robust or
  decision-focused targets). Inventory, sales, expiry and lost sales follow the
  age-structured balance; the plan must respect shelf capacity.
* ``scenario`` (ScenPO-SS, Jia et al. 2025): one routing-and-delivery plan for the
  whole lookahead, evaluated against S demand scenarios. Each scenario has its own
  sales, expiry, lost sales and an excess-disposal slack that gives complete recourse
  when realised stock would exceed shelf capacity.

Routing uses a single-commodity-flow (Gavish-Graves) vehicle-flow formulation:
binary arc variables, at most K routes per day, each store visited at most once a
day, and arc flows bounded by vehicle capacity Q (which also eliminates subtours
that carry load). The day-0 routes of every implemented plan are re-optimised
exactly afterwards by subset enumeration (``SubsetRouting.pack``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .simulator import CostParams


@dataclass
class PlanResult:
    q: np.ndarray                 # (N, H) planned deliveries
    visits: np.ndarray            # (N, H) planned visit indicators
    objective: float
    status: int
    success: bool
    mip_gap: float
    dual_bound: float
    seconds: float
    route_cost_plan: float
    info: dict = field(default_factory=dict)


class _Builder:
    def __init__(self):
        self.nv = 0
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.c: list[float] = []
        self.integ: list[int] = []
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.bl: list[float] = []
        self.bu: list[float] = []
        self.nr = 0

    def var(self, n: int, lb=0.0, ub=np.inf, cost=0.0, integer=False) -> np.ndarray:
        idx = np.arange(self.nv, self.nv + n)
        self.nv += n
        self.lb += [lb] * n if np.isscalar(lb) else list(lb)
        self.ub += [ub] * n if np.isscalar(ub) else list(ub)
        self.c += [cost] * n if np.isscalar(cost) else list(cost)
        self.integ += [1 if integer else 0] * n
        return idx

    def row(self, cols, vals, lo, hi) -> int:
        r = self.nr
        self.nr += 1
        self.rows += [r] * len(cols)
        self.cols += list(int(c) for c in cols)
        self.vals += list(float(v) for v in vals)
        self.bl.append(lo)
        self.bu.append(hi)
        return r


class PIRPModel:
    """Template MILP for a fixed network, horizon, shelf life and scenario count."""

    def __init__(self, dist: np.ndarray, store_cap: np.ndarray, n_vehicles: int, vehicle_cap: float,
                 shelf_life: int, horizon: int, costs: CostParams, n_scen: int = 1,
                 relax_routing: bool = False, routing: str = "assign", tour_cost: np.ndarray | None = None,
                 integer_days: int | None = None):
        self.dist = dist
        self.N = dist.shape[0] - 1
        self.C = np.asarray(store_cap, dtype=float)
        self.K = n_vehicles
        self.Q = float(vehicle_cap)
        self.m = shelf_life
        self.H = horizon
        self.costs = costs
        self.S = n_scen
        self.relax = relax_routing
        # routing binaries only on the first `integer_days` days (None = all days)
        self.integer_days = horizon if integer_days is None else integer_days
        self.routing = routing
        self.tour_cost = tour_cost
        self._build()

    # ------------------------------------------------------------------ build
    def _build_route_block(self, b: "_Builder") -> None:
        """Set-partitioning routing: one binary per (route = store subset, day).

        Route costs are exact TSP costs of each subset (Held-Karp). Per-route delivery
        variables enforce the vehicle capacity of each route exactly.
        """
        N, H = self.N, self.H
        if self.tour_cost is None:
            from .network import held_karp_all_subsets
            self.tour_cost = held_karp_all_subsets(self.dist)
        masks = np.arange(1, 1 << N)
        self.route_masks = masks
        members = [[i for i in range(N) if (mk >> i) & 1] for mk in masks]
        qmax = np.minimum(self.C, self.Q)
        self.y = np.zeros((H, len(masks)), dtype=int)
        self.qv = np.zeros((H, N), dtype=int)
        for k in range(H):
            self.y[k] = b.var(len(masks), 0, 1, self.costs.c_route * self.tour_cost[masks], integer=not self.relax)
            self.qv[k] = b.var(N, 0, qmax, 0.0)
            g_of_store: list[list[int]] = [[] for _ in range(N)]
            routes_of_store: list[list[int]] = [[] for _ in range(N)]
            for r, mem in enumerate(members):
                g = b.var(len(mem), 0, np.inf, 0.0)
                b.row(list(g) + [self.y[k][r]], [1] * len(mem) + [-self.Q], -np.inf, 0)
                for i, gi in zip(mem, g):
                    g_of_store[i].append(gi)
                    routes_of_store[i].append(self.y[k][r])
            b.row(list(self.y[k]), [1] * len(masks), -np.inf, self.K)
            for i in range(N):
                b.row(routes_of_store[i], [1] * len(routes_of_store[i]), -np.inf, 1)
                b.row([self.qv[k][i]] + g_of_store[i], [1] + [-1] * len(g_of_store[i]), 0, 0)
                b.row([self.qv[k][i]] + routes_of_store[i], [1] + [-qmax[i]] * len(routes_of_store[i]), -np.inf, 0)
        self._members = members

    def _build_assign_block(self, b: "_Builder") -> None:
        """Vehicle-assignment routing with exact subset-TSP route costs.

        Binary z[i,v,k] assigns store i to vehicle v on day k; per-vehicle loads respect
        Q. Continuous y[R,v,k] >= 0 select one route (store subset R) per vehicle with
        sum_R y = 1 (the empty route has zero cost) and sum_{R ni i} y = z[i,v,k]. For any
        integral z the only feasible y puts weight one on the assigned subset, so the
        routing cost equals the exact TSP cost of every vehicle's stores; only z is branched.
        """
        N, H, K = self.N, self.H, self.K
        if self.tour_cost is None:
            from .network import held_karp_all_subsets
            self.tour_cost = held_karp_all_subsets(self.dist)
        masks = np.arange(0, 1 << N)
        self.route_masks = masks
        members = [[i for i in range(N) if (mk >> i) & 1] for mk in masks]
        self._members = members
        qmax = np.minimum(self.C, self.Q)
        self.qv = np.zeros((H, N), dtype=int)
        self.z = np.zeros((H, K, N), dtype=int)
        self.qkv = np.zeros((H, K, N), dtype=int)
        self.yv = np.zeros((H, K, len(masks)), dtype=int)
        for k in range(H):
            self.qv[k] = b.var(N, 0, qmax, 0.0)
            for v in range(K):
                self.z[k, v] = b.var(N, 0, 1, 0.0, integer=(not self.relax) and k < self.integer_days)
                self.qkv[k, v] = b.var(N, 0, qmax, 0.0)
                self.yv[k, v] = b.var(len(masks), 0, 1, self.costs.c_route * self.tour_cost[masks])
            for v in range(K):
                y = self.yv[k, v]
                b.row(list(y), [1] * len(masks), 1, 1)
                for i in range(N):
                    cols = [y[r] for r, mem in enumerate(members) if i in mem]
                    b.row(cols + [self.z[k, v, i]], [1] * len(cols) + [-1], 0, 0)
                    b.row([self.qkv[k, v, i], self.z[k, v, i]], [1, -qmax[i]], -np.inf, 0)
                b.row(list(self.qkv[k, v]), [1] * N, -np.inf, self.Q)
            for i in range(N):
                b.row([self.z[k, v, i] for v in range(K)], [1] * K, -np.inf, 1)
                b.row([self.qv[k][i]] + [self.qkv[k, v, i] for v in range(K)], [1] + [-1] * K, 0, 0)
                # symmetry breaking: store i may use vehicle v only if a lower store uses v-1
                for v in range(1, K):
                    lower = [self.z[k, v - 1, j] for j in range(i)]
                    b.row([self.z[k, v, i]] + lower, [1] + [-1] * len(lower), -np.inf, 0)

    def _build(self) -> None:
        N, H, m, S = self.N, self.H, self.m, self.S
        b = _Builder()
        if self.routing == "assign":
            self._build_assign_block(b)
            self._build_inventory_block(b)
            return
        if self.routing == "routes":
            self._build_route_block(b)
            self._build_inventory_block(b)
            return
        nodes = N + 1
        arcs = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
        self.arcs = arcs
        A = len(arcs)
        arc_cost = np.array([self.costs.c_route * self.dist[i, j] for i, j in arcs])
        qmax = np.minimum(self.C, self.Q)
        self.x = np.zeros((H, A), dtype=int)
        self.f = np.zeros((H, A), dtype=int)
        self.qv = np.zeros((H, N), dtype=int)
        for k in range(H):
            self.x[k] = b.var(A, 0, 1, arc_cost, integer=not self.relax)
            ubf = np.array([0.0 if j == 0 else self.Q for i, j in arcs])
            self.f[k] = b.var(A, 0, ubf, 0.0)
            self.qv[k] = b.var(N, 0, qmax, 0.0)
        out_arcs = {i: [a for a, (u, v) in enumerate(arcs) if u == i] for i in range(nodes)}
        in_arcs = {i: [a for a, (u, v) in enumerate(arcs) if v == i] for i in range(nodes)}
        self._out_arcs = out_arcs
        for k in range(H):
            xk, fk = self.x[k], self.f[k]
            dep_out = [xk[a] for a in out_arcs[0]]
            dep_in = [xk[a] for a in in_arcs[0]]
            b.row(dep_out, [1] * len(dep_out), -np.inf, self.K)
            b.row(dep_out + dep_in, [1] * len(dep_out) + [-1] * len(dep_in), 0, 0)
            for i in range(1, nodes):
                oi = [xk[a] for a in out_arcs[i]]
                ii = [xk[a] for a in in_arcs[i]]
                b.row(oi, [1] * len(oi), -np.inf, 1)
                b.row(oi + ii, [1] * len(oi) + [-1] * len(ii), 0, 0)
                # delivery only if visited
                b.row([self.qv[k][i - 1]] + oi, [1] + [-qmax[i - 1]] * len(oi), -np.inf, 0)
                # flow conservation: inflow - outflow = delivery
                fin = [fk[a] for a in in_arcs[i]]
                fout = [fk[a] for a in out_arcs[i]]
                b.row(fin + fout + [self.qv[k][i - 1]], [1] * len(fin) + [-1] * len(fout) + [-1], 0, 0)
            for a in range(A):
                b.row([fk[a], xk[a]], [1, -self.Q], -np.inf, 0)
            # no 2-cycles between stores
            for i in range(1, nodes):
                for j in range(i + 1, nodes):
                    a1 = arcs.index((i, j))
                    a2 = arcs.index((j, i))
                    b.row([xk[a1], xk[a2]], [1, 1], -np.inf, 1)
        self._build_inventory_block(b)

    def _build_inventory_block(self, b: "_Builder") -> None:
        N, H, m, S = self.N, self.H, self.m, self.S
        h, ps, pu = self.costs.h, self.costs.p_waste, self.costs.p_lost
        w_s = 1.0 / S
        self.dem_rows = np.zeros((S, H, N), dtype=int)
        self.init_rows = np.zeros((S, N, max(m - 1, 1)), dtype=int)
        self.s_idx = np.zeros((S, H, N, m), dtype=int)
        self.w_idx = np.zeros((S, H, N, m), dtype=int)
        self.u_idx = np.zeros((S, H, N), dtype=int)
        self.z_idx = np.zeros((S, H, N, m), dtype=int)
        for s in range(S):
            for k in range(H):
                # objective on s and w encodes holding (r>=2 carried) and expiry (r=1 unsold)
                cs = np.array([ps] + [h] * (m - 1)) * w_s
                for i in range(N):
                    self.s_idx[s, k, i] = b.var(m, 0, np.inf, cs)
                    self.w_idx[s, k, i] = b.var(m, 0, np.inf, -cs)
                    self.u_idx[s, k, i] = b.var(1, 0, np.inf, pu * w_s)[0]
                    if S > 1:
                        # disposal of excess stock (complete recourse), counted as waste
                        self.z_idx[s, k, i] = b.var(m, 0, np.inf, np.array([0.0] + [ps - h] * (m - 1)) * w_s)
            for k in range(H):
                for i in range(N):
                    sv, wv, uv = self.s_idx[s, k, i], self.w_idx[s, k, i], self.u_idx[s, k, i]
                    zv = self.z_idx[s, k, i] if S > 1 else None
                    self.dem_rows[s, k, i] = b.row(list(wv) + [uv], [1] * (m + 1), 0, 0)
                    for r in range(m):
                        cols, vals = [wv[r], sv[r]], [1, -1]
                        if zv is not None:
                            cols.append(zv[r]); vals.append(1)
                        b.row(cols, vals, -np.inf, 0)
                    # fresh units equal the delivery
                    b.row([sv[m - 1], self.qv[k][i]], [1, -1], 0, 0)
                    # shelf capacity on stock available at opening
                    cols, vals = list(sv), [1] * m
                    if zv is not None:
                        cols += list(zv); vals += [-1] * m
                    b.row(cols, vals, -np.inf, self.C[i])
                    if k == 0:
                        for r in range(m - 1):
                            self.init_rows[s, i, r] = b.row([sv[r]], [1], 0, 0)
                    else:
                        sp, wp = self.s_idx[s, k - 1, i], self.w_idx[s, k - 1, i]
                        zp = self.z_idx[s, k - 1, i] if S > 1 else None
                        for r in range(m - 1):
                            cols, vals = [sv[r], sp[r + 1], wp[r + 1]], [1, -1, 1]
                            if zp is not None:
                                cols.append(zp[r + 1]); vals.append(1)
                            b.row(cols, vals, 0, 0)
        self.nv = b.nv
        self.c = np.array(b.c)
        self.lb = np.array(b.lb)
        self.ub = np.array(b.ub)
        self.integrality = np.array(b.integ)
        self.A = coo_matrix((b.vals, (b.rows, b.cols)), shape=(b.nr, b.nv)).tocsr()
        self.bl0 = np.array(b.bl, dtype=float)
        self.bu0 = np.array(b.bu, dtype=float)

    # ------------------------------------------------------------------ solve
    def solve(self, B0: np.ndarray, demand: np.ndarray, time_limit: float = 20.0,
              mip_rel_gap: float = 0.005) -> PlanResult:
        """B0: (N, m-1) state; demand: (N, H) or (S, N, H) planning demand."""
        demand = np.asarray(demand, dtype=float)
        if demand.ndim == 2:
            demand = demand[None]
        assert demand.shape == (self.S, self.N, self.H)
        bl, bu = self.bl0.copy(), self.bu0.copy()
        for s in range(self.S):
            rows = self.dem_rows[s].reshape(self.H, self.N)
            vals = np.maximum(demand[s].T, 0.0)
            bl[rows] = vals
            bu[rows] = vals
            if self.m > 1:
                ir = self.init_rows[s][:, : self.m - 1]
                bl[ir] = B0
                bu[ir] = B0
        t0 = time.perf_counter()
        res = milp(self.c, integrality=self.integrality, bounds=Bounds(self.lb, self.ub),
                   constraints=LinearConstraint(self.A, bl, bu),
                   options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
        secs = time.perf_counter() - t0
        if res.x is None:
            return PlanResult(np.zeros((self.N, self.H)), np.zeros((self.N, self.H)), np.inf, res.status,
                              False, np.inf, np.nan, secs, np.nan, {"message": res.message})
        x = res.x
        q = np.array([[x[self.qv[k][i]] for k in range(self.H)] for i in range(self.N)])
        q[q < 1e-7] = 0.0
        visits = np.zeros((self.N, self.H))
        rc = 0.0
        routes = []
        for k in range(self.H):
            if self.routing == "assign":
                day_routes = []
                for v in range(self.K):
                    ys = x[self.yv[k, v]]
                    zs = x[self.z[k, v]]
                    rc += float(np.dot(ys, self.costs.c_route * self.tour_cost[self.route_masks]))
                    visits[:, k] += zs
                    mk = int(sum(1 << i for i in range(self.N) if zs[i] > 0.5))
                    if mk:
                        day_routes.append(mk)
                routes.append(day_routes)
                continue
            if self.routing == "routes":
                ys = x[self.y[k]]
                rc += float(np.dot(ys, self.costs.c_route * self.tour_cost[self.route_masks]))
                chosen = np.where(ys > 0.5)[0]
                routes.append([int(self.route_masks[r]) for r in chosen])
                for r in np.where(ys > 1e-9)[0]:
                    for i in self._members[r]:
                        visits[i, k] += ys[r]
                continue
            xs = x[self.x[k]]
            rc += float(np.dot(xs, [self.costs.c_route * self.dist[i, j] for i, j in self.arcs]))
            for i in range(1, self.N + 1):
                visits[i - 1, k] = sum(xs[a] for a in self._out_arcs[i])
        gap = getattr(res, "mip_gap", np.nan)
        dual = getattr(res, "mip_dual_bound", np.nan)
        return PlanResult(q, np.round(visits, 6), float(res.fun), res.status, res.status == 0,
                          float(gap) if gap is not None else np.nan,
                          float(dual) if dual is not None else np.nan, secs, rc,
                          {"message": res.message, "routes": routes})


def recourse_lp_duals(B0: np.ndarray, Q: np.ndarray, D: np.ndarray, m: int, costs: CostParams):
    """Solve the inventory recourse LP for fixed deliveries; return (value, dual wrt deliveries).

    Used to verify Theorem 1 (LP recourse value equals the FIFO simulation cost) and
    Proposition 2 (dual multipliers of the fresh-stock rows are subgradients).
    Shelf capacity is not imposed here because deliveries are fixed and feasible.
    """
    from scipy.optimize import linprog

    N, H = Q.shape
    b = _Builder()
    h, ps, pu = costs.h, costs.p_waste, costs.p_lost
    s_idx = np.zeros((H, N, m), dtype=int)
    w_idx = np.zeros((H, N, m), dtype=int)
    u_idx = np.zeros((H, N), dtype=int)
    fresh_rows = np.zeros((H, N), dtype=int)
    for k in range(H):
        cs = np.array([ps] + [h] * (m - 1))
        for i in range(N):
            s_idx[k, i] = b.var(m, 0, np.inf, cs)
            w_idx[k, i] = b.var(m, 0, np.inf, -cs)
            u_idx[k, i] = b.var(1, 0, np.inf, pu)[0]
    for k in range(H):
        for i in range(N):
            sv, wv, uv = s_idx[k, i], w_idx[k, i], u_idx[k, i]
            b.row(list(wv) + [uv], [1] * (m + 1), D[i, k], D[i, k])
            for r in range(m):
                b.row([wv[r], sv[r]], [1, -1], -np.inf, 0)
            fresh_rows[k, i] = b.row([sv[m - 1]], [1], Q[i, k], Q[i, k])
            if k == 0:
                for r in range(m - 1):
                    b.row([sv[r]], [1], B0[i, r], B0[i, r])
            else:
                sp, wp = s_idx[k - 1, i], w_idx[k - 1, i]
                for r in range(m - 1):
                    b.row([sv[r], sp[r + 1], wp[r + 1]], [1, -1, 1], 0, 0)
    A = coo_matrix((b.vals, (b.rows, b.cols)), shape=(b.nr, b.nv)).tocsr()
    bl, bu = np.array(b.bl), np.array(b.bu)
    eq = bl == bu
    res = linprog(np.array(b.c), A_ub=A[~eq], b_ub=bu[~eq], A_eq=A[eq], b_eq=bl[eq],
                  bounds=list(zip(b.lb, b.ub)), method="highs")
    eq_index = np.cumsum(eq) - 1
    duals = res.eqlin.marginals
    pi = np.array([[duals[eq_index[fresh_rows[k, i]]] for k in range(H)] for i in range(N)])
    # the recourse objective omits the constant carried-holding of initial stock; add it for parity
    return float(res.fun), pi
