"""Replenishment-and-routing policies evaluated in closed loop.

Every policy is called once per day with the same information: the observed
age-structured store inventory, the day's forecasts for the next H days, and the
episode-start network data. Only the perfect-information reference also sees the
realised future demand. Each policy returns today's delivery quantities; the
runner then routes them with the exact CVRP oracle and applies the FIFO dynamics.

Families
--------
Inventory-control rules, route-second (decoupled):
  * BSP     periodic base-stock on total stock (inventory position ignores age)
  * BSP-EW  base-stock raised by the estimated waste of current stock within the
            review window (Broekmeulen & van Donselaar 2009; Haijema & Minner 2019)
  * UL      Crama et al. (2018) deliver-up-to-level rule UL_lambda
Integrated lookahead (rolling-horizon PIRP MILP):
  * PTO-mean, PTO-quantile (cost-function approximation), Robust (budgeted
    uncertainty, Bertsimas & Thiele 2006), ScenPO-SS (Jia et al. 2025),
    DFL (decision-focused planning targets), PI (perfect-information reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .forecasting import QUANTILE_LEVELS, quantile_at, sigma_from_quantiles
from .milp import PIRPModel
from .network import SubsetRouting
from .simulator import CostParams


@dataclass
class Instance:
    dist: np.ndarray
    store_cap: np.ndarray
    n_vehicles: int
    vehicle_cap: float
    shelf_life: int
    horizon: int
    costs: CostParams
    tour_cost: np.ndarray | None = None
    lifo_share: float = 0.0

    @property
    def N(self) -> int:
        return self.dist.shape[0] - 1


@dataclass
class Context:
    day: int
    B: np.ndarray            # (N, m-1) inventory by remaining life at opening (before delivery)
    mu: np.ndarray           # (N, H) mean forecasts
    qs: np.ndarray           # (N, H, L) quantile forecasts
    X: np.ndarray            # (N*H, F) features of the forecast rows
    future: np.ndarray | None = None  # (N, H) realised demand, PI reference only
    seed: int = 0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def cap_room(inst: Instance, B: np.ndarray) -> np.ndarray:
    return np.maximum(inst.store_cap - B.sum(axis=1), 0.0)


def repair(q: np.ndarray, urgent: np.ndarray, router: SubsetRouting, inst: Instance, B: np.ndarray) -> np.ndarray:
    """Make a rule-based delivery vector feasible: shelf room, fleet capacity and packability.

    The urgent part (today's need) is protected; any excess is scaled down first.
    """
    q = np.minimum(np.maximum(q, 0.0), cap_room(inst, B))
    urgent = np.minimum(urgent, q)
    if router.pack(q) is not None:
        return q
    extra = q - urgent
    lo, hi = 0.0, 1.0
    if router.pack(urgent) is not None:
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if router.pack(urgent + mid * extra) is not None:
                lo = mid
            else:
                hi = mid
        return urgent + lo * extra
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if router.pack(mid * urgent) is not None:
            lo = mid
        else:
            hi = mid
    return lo * urgent


def projected_waste(B: np.ndarray, mu: np.ndarray, window: int) -> np.ndarray:
    """Units of current stock expected to expire unsold within `window` days (FIFO, mean demand)."""
    N, m1 = B.shape
    stock = B.copy().astype(float)
    waste = np.zeros(N)
    for k in range(min(window, m1)):
        rem = mu[:, k].copy() if k < mu.shape[1] else mu[:, -1].copy()
        for r in range(stock.shape[1]):
            take = np.minimum(stock[:, r], rem)
            stock[:, r] -= take
            rem -= take
        waste += stock[:, 0]
        stock = np.concatenate([stock[:, 1:], np.zeros((N, 1))], axis=1)
    return waste


def cumulative_quantile(mu: np.ndarray, sig: np.ndarray, qs: np.ndarray, days: int, alpha: float) -> np.ndarray:
    """alpha-quantile of demand summed over the first `days` days (normal approximation for days>1)."""
    if days == 1:
        return quantile_at(qs[:, 0], alpha)
    m = mu[:, :days].sum(1)
    s = np.sqrt((sig[:, :days] ** 2).sum(1))
    return np.maximum(m + norm.ppf(alpha) * s, 0.0)


# --------------------------------------------------------------------------- rule policies
class BaseStockPolicy:
    """Periodic base-stock with delivery cycle R (stores split into R staggered groups)."""

    def __init__(self, inst: Instance, router: SubsetRouting, R: int = 1, alpha: float = 0.9,
                 waste_aware: bool = False):
        self.inst, self.router, self.R, self.alpha, self.ew = inst, router, R, alpha, waste_aware
        self.name = ("BSP-EW" if waste_aware else "BSP") + f"(R={R},a={alpha})"

    def decide(self, ctx: Context) -> tuple[np.ndarray, dict]:
        N = self.inst.N
        sig = sigma_from_quantiles(ctx.qs)
        S = cumulative_quantile(ctx.mu, sig, ctx.qs, self.R, self.alpha)
        ip = ctx.B.sum(1)
        if self.ew:
            ip = ip - projected_waste(ctx.B, ctx.mu, self.R)
        q = np.maximum(S - ip, 0.0)
        eligible = (np.arange(N) % self.R) == (ctx.day % self.R)
        q = np.where(eligible, q, 0.0)
        urgent = np.maximum(quantile_at(ctx.qs[:, 0], 0.5) - ctx.B.sum(1), 0.0)
        return repair(q, urgent, self.router, self.inst, ctx.B), {}


class ULPolicy:
    """Crama et al. (2018) UL_lambda: visit a store when its stock does not cover today's demand
    at service level beta; then deliver up to the alpha-quantile of lambda days of demand."""

    def __init__(self, inst: Instance, router: SubsetRouting, lam: int = 2, alpha: float = 0.9, beta: float = 0.7):
        self.inst, self.router, self.lam, self.alpha, self.beta = inst, router, lam, alpha, beta
        self.name = f"UL(lam={lam},a={alpha},b={beta})"

    def decide(self, ctx: Context) -> tuple[np.ndarray, dict]:
        sig = sigma_from_quantiles(ctx.qs)
        stock = ctx.B.sum(1)
        trigger = stock < quantile_at(ctx.qs[:, 0], self.beta)
        lam = min(self.lam, ctx.mu.shape[1])
        S = cumulative_quantile(ctx.mu, sig, ctx.qs, lam, self.alpha)
        usable = stock - projected_waste(ctx.B, ctx.mu, lam)
        q = np.where(trigger, np.maximum(S - usable, 0.0), 0.0)
        urgent = np.where(trigger, np.maximum(quantile_at(ctx.qs[:, 0], self.beta) - stock, 0.0), 0.0)
        return repair(q, urgent, self.router, self.inst, ctx.B), {}


# --------------------------------------------------------------------------- lookahead policies
class LookaheadPolicy:
    """Rolling-horizon PIRP: solve the H-day MILP for a planning-demand input, implement day 0."""

    def __init__(self, inst: Instance, router: SubsetRouting, target: str, param: float = 0.0, n_scen: int = 1,
                 time_limit: float = 20.0, gap: float = 0.005, dfl_model=None, name: str | None = None,
                 integer_days: int | None = None):
        self.inst, self.router, self.target, self.param = inst, router, target, param
        self.n_scen = n_scen if target == "scenario" else 1
        self.time_limit, self.gap = time_limit, gap
        self.dfl_model = dfl_model
        self.integer_days = integer_days
        self.name = name or f"{target}({param})"
        self._model: PIRPModel | None = None

    @property
    def model(self) -> PIRPModel:
        if self._model is None:
            i = self.inst
            self._model = PIRPModel(i.dist, i.store_cap, i.n_vehicles, i.vehicle_cap, i.shelf_life, i.horizon,
                                    i.costs, n_scen=self.n_scen, tour_cost=i.tour_cost,
                                    integer_days=self.integer_days)
        return self._model

    def planning_demand(self, ctx: Context) -> np.ndarray:
        if self.target == "mean":
            return ctx.mu
        if self.target == "quantile":
            return quantile_at(ctx.qs, self.param)
        if self.target == "robust":
            return robust_targets(ctx.mu, ctx.qs, self.param)
        if self.target == "scenario":
            return sample_scenarios(ctx.qs, self.n_scen, ctx.seed)
        if self.target == "pi":
            return ctx.future
        if self.target == "dfl":
            return self.dfl_model.targets(ctx)
        raise ValueError(self.target)

    def decide(self, ctx: Context) -> tuple[np.ndarray, dict]:
        d = self.planning_demand(ctx)
        res = self.model.solve(ctx.B, d, time_limit=self.time_limit, mip_rel_gap=self.gap)
        info = {"status": res.status, "gap": res.mip_gap, "secs": res.seconds, "success": res.success,
                "visits": res.visits, "plan_q": res.q}
        if not np.isfinite(res.objective):
            # no incumbent inside the time limit: fall back to the myopic quantile rule (recorded)
            info["fallback"] = True
            q = np.minimum(np.maximum(d if d.ndim == 2 else d.mean(0), 0)[:, 0] - ctx.B.sum(1), cap_room(self.inst, ctx.B))
            return repair(np.maximum(q, 0), np.maximum(q, 0), self.router, self.inst, ctx.B), info
        q0 = np.minimum(res.q[:, 0], cap_room(self.inst, ctx.B))
        return q0, info


def robust_targets(mu: np.ndarray, qs: np.ndarray, gamma: float) -> np.ndarray:
    """Budgeted-uncertainty planning demand (Bertsimas & Thiele 2006).

    Demand on day k lies in [mu_k, mu_k + dhat_k] with dhat the 95%-quantile excess; the
    cumulative scaled deviation through day k is at most Gamma_k = min(gamma*sqrt(k+1), k+1).
    The planning demand reproduces the worst-case cumulative demand of every prefix.
    """
    i95 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.95))[0][0])
    dhat = np.maximum(qs[..., i95] - mu, 0.0)
    N, H = mu.shape
    worst = np.zeros((N, H))
    for k in range(H):
        g = min(gamma * np.sqrt(k + 1), k + 1)
        devs = -np.sort(-dhat[:, :k + 1], axis=1)
        full = int(np.floor(g))
        frac = g - full
        add = devs[:, :full].sum(1)
        if full < k + 1:
            add = add + frac * devs[:, full]
        worst[:, k] = mu[:, :k + 1].sum(1) + add
    out = np.diff(np.concatenate([np.zeros((N, 1)), worst], axis=1), axis=1)
    return np.maximum(out, 0.0)


def sample_scenarios(qs: np.ndarray, n_scen: int, seed: int) -> np.ndarray:
    """Jia et al. (2025) scenario sampling from multi-horizon quantile forecasts.

    For each scenario a quantile band [tau_b, tau_{b+1}] is drawn uniformly; every
    (store, day) value is then drawn uniformly between the two predicted quantiles.
    """
    rng = np.random.default_rng(seed)
    N, H, L = qs.shape
    out = np.empty((n_scen, N, H))
    for s in range(n_scen):
        b = rng.integers(0, L - 1)
        u = rng.random((N, H))
        out[s] = qs[..., b] + u * (qs[..., b + 1] - qs[..., b])
    return out
