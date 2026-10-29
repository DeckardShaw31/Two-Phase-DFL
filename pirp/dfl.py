"""Decision-focused learning (DFL) of planning targets for the rolling-horizon PIRP.

Planning target for store i and lookahead day k:

    d_tilde[i,k] = q_alpha*[i,k] + sigma_hat[i,k] * delta_theta(x[i,k], state_i)

where q_alpha* is the predicted quantile at the level selected for the quantile
lookahead policy on validation data, sigma_hat a forecast scale and delta_theta a
small network (zero-initialised, so training starts exactly at the tuned quantile
policy). The network sees the forecast features and the store's age-structured
inventory, so the safety buffer can depend on context, coverage length and stock age.

Training signal. For a fixed routing/visit pattern chosen by the MILP oracle at the
current targets, deliveries are a piecewise-linear function of the targets
(``coverage_quantities``: each visit covers planned demand until the next visit or
the end of shelf life, net of planned usable stock, capped by shelf and vehicle
capacity). The realised H-day inventory cost of those deliveries is evaluated by the
FIFO simulator under the true demand (``torch_fifo_cost``). Autograd gives the
chain d cost/d q (an optimal dual of the recourse LP, Proposition 2) times
d q/d d_tilde (the local response of the plan with routes fixed). Visit patterns are
re-optimised with the MILP every ``resolve_every`` epochs (solution caching).
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn

from .forecasting import quantile_at, sigma_from_quantiles
from .policies import Context, Instance
from .simulator import torch_fifo_cost

torch.set_num_threads(1)


def row_features(ctx_X: np.ndarray, mu: np.ndarray, qs: np.ndarray, B: np.ndarray, mean_store: np.ndarray,
                 H: int) -> np.ndarray:
    """Per (store, k) feature rows: forecast features + scaled forecasts + scaled inventory state."""
    N = mu.shape[0]
    sig = sigma_from_quantiles(qs)
    scale = mean_store[:, None]
    stock = B / mean_store[:, None]
    rows = []
    for i in range(N):
        for k in range(H):
            koh = np.zeros(H)
            koh[k] = 1.0
            rows.append(np.concatenate([ctx_X[i * H + k, 3:], [mu[i, k] / scale[i, 0], sig[i, k] / scale[i, 0],
                                                               (qs[i, k, -1] - qs[i, k, 0]) / scale[i, 0]],
                                        stock[i], [stock[i].sum()], koh]))
    return np.asarray(rows, dtype=np.float32)


class DeltaNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class DFLTargetModel:
    def __init__(self, alpha: float, mean_store: np.ndarray, H: int, n_in: int, x_mean=None, x_std=None,
                 hidden: int = 64):
        self.alpha, self.mean_store, self.H = alpha, np.asarray(mean_store), H
        self.net = DeltaNet(n_in, hidden)
        self.x_mean = np.zeros(n_in, np.float32) if x_mean is None else x_mean
        self.x_std = np.ones(n_in, np.float32) if x_std is None else x_std
        self.hidden = hidden

    # -- (de)serialisation for worker processes
    def state(self) -> dict:
        return {"alpha": self.alpha, "mean_store": self.mean_store, "H": self.H, "n_in": len(self.x_mean),
                "x_mean": self.x_mean, "x_std": self.x_std, "hidden": self.hidden,
                "weights": {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}}

    @classmethod
    def from_state(cls, st: dict) -> "DFLTargetModel":
        m = cls(st["alpha"], st["mean_store"], st["H"], st["n_in"], st["x_mean"], st["x_std"], st["hidden"])
        m.net.load_state_dict({k: torch.tensor(v) for k, v in st["weights"].items()})
        m.net.eval()
        return m

    def torch_targets(self, feats: torch.Tensor, base: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        x = (feats - torch.as_tensor(self.x_mean)) / torch.as_tensor(self.x_std)
        return torch.clamp(base + sig * self.net(x), min=0.0)

    def targets(self, ctx: Context) -> np.ndarray:
        N, H = ctx.mu.shape
        feats = row_features(ctx.X, ctx.mu, ctx.qs, ctx.B, self.mean_store, H)
        base = quantile_at(ctx.qs, self.alpha).reshape(-1)
        sig = sigma_from_quantiles(ctx.qs).reshape(-1)
        # only the correction comes from the float32 network; the targets are assembled in
        # float64 so that a zero correction reproduces the quantile policy bit for bit (the
        # MILP incumbent within the 0.5% gap is sensitive to float32 rounding of its inputs)
        with torch.no_grad():
            x = (torch.from_numpy(feats) - torch.as_tensor(self.x_mean)) / torch.as_tensor(self.x_std)
            delta = self.net(x).numpy().astype(float)
        return np.maximum(base + sig * delta, 0.0).reshape(N, H)


def coverage_quantities(d: torch.Tensor, visits: np.ndarray, vehicle_of: np.ndarray, B0: torch.Tensor,
                        cap: torch.Tensor, Q: float, m: int) -> torch.Tensor:
    """Deliveries implied by planning demand d (N,H) for a fixed visit pattern.

    visits: (N,H) 0/1; vehicle_of: (N,H) vehicle index or -1. Each delivery covers the planned
    demand from its day until the next planned visit, limited by shelf life and horizon, net of
    what the planned stock can still serve in that window (FIFO); it is capped by shelf room and
    scaled down proportionally if the vehicle load would exceed Q.
    """
    N, H = d.shape
    stock = B0.clone()  # (N, m-1) remaining-life buckets
    qcols = []
    for k in range(H):
        vk = torch.as_tensor(visits[:, k], dtype=d.dtype)
        nxt = np.full(N, H)
        for i in range(N):
            later = np.where(visits[i, k + 1:] > 0.5)[0]
            if len(later):
                nxt[i] = k + 1 + later[0]
        end = np.minimum(nxt, k + m)
        end = np.minimum(end, H)
        # demand of the coverage window
        mask = torch.zeros(N, H, dtype=d.dtype)
        for i in range(N):
            mask[i, k:end[i]] = 1.0
        window_dem = (d * mask).sum(1)
        # what existing stock can serve inside the window (FIFO, planned demand)
        s = stock.clone()
        served = torch.zeros(N, dtype=d.dtype)
        for j in range(k, H):
            active = torch.as_tensor((j < end).astype(float), dtype=d.dtype)
            rem = d[:, j] * active
            cols = []
            for r in range(s.shape[1]):
                take = torch.minimum(s[:, r], rem)
                served = served + take
                rem = rem - take
                cols.append(s[:, r] - take)
            s = torch.stack(cols[1:] + [torch.zeros(N, dtype=d.dtype)], dim=1) if s.shape[1] > 0 else s
        room = torch.clamp(cap - stock.sum(1), min=0.0)
        q = torch.minimum(torch.clamp(window_dem - served, min=0.0), room) * vk
        # vehicle capacity
        veh = vehicle_of[:, k]
        for v in np.unique(veh[veh >= 0]):
            idx = torch.as_tensor(np.where(veh == v)[0])
            load = q[idx].sum()
            factor = torch.clamp(Q / (load + 1e-9), max=1.0)
            q = q.index_put((idx,), q[idx] * factor)
        qcols.append(q)
        # advance planned stock one day with planned demand
        avail = torch.cat([stock, q[:, None]], dim=1)
        rem = d[:, k]
        cols = []
        for r in range(avail.shape[1]):
            take = torch.minimum(avail[:, r], rem)
            cols.append(avail[:, r] - take)
            rem = rem - take
        stock = torch.stack(cols[1:], dim=1)
    return torch.stack(qcols, dim=1)


# ---------------------------------------------------------------------------- oracle calls
_ORACLE: dict = {}


def _oracle_worker(args):
    inst, B0, d, integer_days = args
    from .milp import PIRPModel
    _ORACLE["integer_days"] = integer_days
    key = (inst.shelf_life, inst.horizon, inst.vehicle_cap, tuple(inst.store_cap), integer_days)
    if _ORACLE.get("key") != key:
        _ORACLE["model"] = PIRPModel(inst.dist, inst.store_cap, inst.n_vehicles, inst.vehicle_cap,
                                     inst.shelf_life, inst.horizon, inst.costs, tour_cost=inst.tour_cost,
                                     integer_days=_ORACLE.get("integer_days"))
        _ORACLE["key"] = key
    res = _ORACLE["model"].solve(B0, d, time_limit=10.0, mip_rel_gap=0.01)
    N, H = d.shape
    veh = -np.ones((N, H), dtype=int)
    routes = res.info.get("routes", [[] for _ in range(H)])
    for k, masks in enumerate(routes):
        for v, mk in enumerate(masks):
            for i in range(N):
                if (mk >> i) & 1:
                    veh[i, k] = v
    # a store is visited on a lookahead day when the plan delivers to it (on LP-relaxed days the
    # assignment can be fractional, so planned deliveries define the pattern)
    visits = ((veh >= 0) | (res.q > 1e-6)).astype(float)
    return visits, veh, res.q, res.route_cost_plan, res.status


def solve_patterns(inst: Instance, samples: list[dict], targets: list[np.ndarray], pool: ProcessPoolExecutor,
                   integer_days: int | None = None):
    args = [(inst, s["B"], t, integer_days) for s, t in zip(samples, targets)]
    return list(pool.map(_oracle_worker, args, chunksize=4))


# ---------------------------------------------------------------------------- training
def train_dfl(inst: Instance, samples: list[dict], alpha: float, mean_store: np.ndarray, val_fn,
              epochs: int = 8, resolve_every: int = 1, lr: float = 2e-3, batch: int = 64, l2: float = 1e-3,
              n_workers: int = 15, seed: int = 0, log=print,
              integer_days: int | None = None) -> tuple["DFLTargetModel", list[dict]]:
    """samples: dicts with B (N,m-1), X (N*H,F), mu (N,H), qs (N,H,L), d (N,H) true demand.

    val_fn(model_state) -> closed-loop validation cost; the best state (including the
    zero-correction start, i.e. the tuned quantile policy) is returned.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H = inst.horizon
    feats = [row_features(s["X"], s["mu"], s["qs"], s["B"], mean_store, H) for s in samples]
    allf = np.vstack(feats)
    x_mean, x_std = allf.mean(0), allf.std(0) + 1e-6
    model = DFLTargetModel(alpha, mean_store, H, allf.shape[1], x_mean.astype(np.float32), x_std.astype(np.float32))
    base = [torch.from_numpy(quantile_at(s["qs"], alpha).reshape(-1)).float() for s in samples]
    sig = [torch.from_numpy(sigma_from_quantiles(s["qs"]).reshape(-1)).float() for s in samples]
    Ft = [torch.from_numpy(f) for f in feats]
    opt = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=0.0)
    cap = torch.as_tensor(inst.store_cap, dtype=torch.float32)
    N = inst.N
    scale = float(mean_store.sum() * inst.costs.p_lost) / 10.0
    history = []
    best_state = model.state()
    best_val = val_fn(best_state)
    history.append({"epoch": 0, "val_cost": best_val, "train_loss": np.nan})
    log(f"  DFL epoch 0 (tuned quantile start) val={best_val:.2f}")
    patterns = None
    if True:
        for ep in range(1, epochs + 1):
            if patterns is None or (ep - 1) % resolve_every == 0:
                with torch.no_grad():
                    tg = [model.torch_targets(Ft[j], base[j], sig[j]).numpy().reshape(N, H).astype(float)
                          for j in range(len(samples))]
                from .evaluation import get_pool
                patterns = solve_patterns(inst, samples, tg, get_pool(n_workers), integer_days)
            order = rng.permutation(len(samples))
            losses = []
            model.net.train()
            for b0 in range(0, len(order), batch):
                idx = order[b0:b0 + batch]
                opt.zero_grad()
                tot = 0.0
                for j in idx:
                    s = samples[j]
                    visits, veh = patterns[j][0], patterns[j][1]
                    d = model.torch_targets(Ft[j], base[j], sig[j]).reshape(N, H)
                    B0 = torch.as_tensor(s["B"], dtype=torch.float32)
                    q = coverage_quantities(d, visits, veh, B0, cap, inst.vehicle_cap, inst.shelf_life)
                    cost = torch_fifo_cost(B0, q, torch.as_tensor(s["d"], dtype=torch.float32), inst.costs)
                    tot = tot + cost / scale
                loss = tot / len(idx)
                reg = l2 * sum((p ** 2).sum() for p in model.net.head.parameters())
                (loss + reg).backward()
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.0)
                opt.step()
                losses.append(float(loss))
            model.net.eval()
            st = model.state()
            v = val_fn(st)
            history.append({"epoch": ep, "val_cost": v, "train_loss": float(np.mean(losses))})
            log(f"  DFL epoch {ep} train={np.mean(losses):.4f} val={v:.2f}")
            if v < best_val:
                best_val, best_state = v, copy.deepcopy(st)
    return DFLTargetModel.from_state(best_state), history
