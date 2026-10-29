"""Decision-focused learning on top of the deliver-up-to-level rule (UL-DFL).

UL_lambda visits a store when its stock does not cover today's demand at service level beta
and then delivers up to S = Q_alpha(lambda-day demand), net of stock expected to expire.
UL-DFL replaces S by S + s * delta_theta(phi), where s is the forecast scale of the
lambda-day demand and delta_theta a small network of the forecast context and the store's
stock by remaining life. The output layer is zero-initialised and everything is computed in
float64, so training starts exactly at the tuned rule.

theta is trained by back-propagating the realised inventory cost (holding, waste, lost
sales) of whole 21-day training episodes through differentiable FIFO dynamics, with the
visit trigger held fixed (cf. Alvo et al. 2023 for neural ordering policies trained through
simulated inventory dynamics). Fleet capacity enters training as a proportional scaling of
the day's deliveries; exact packing, routing and routing cost enter only the closed-loop
validation used for early stopping.

NN-DFL (ablation) uses the same inputs, differentiable dynamics, loss, seeds and early stopping
but none of the rule's structure: the network outputs the order quantity directly,
q = m * relu(1 + delta_theta(phi)) with m the store's mean demand, a store is visited whenever
q > 0, and training starts from ordering one day of mean demand every day.

NB-DFL (ablation) keeps the order-up-to structure but neither the forecast quantile nor the visit
trigger of the rule: the network outputs the order-up-to level S = m * relu(1 + delta_theta(phi)),
the store receives S minus its usable stock (stock net of units projected to expire today), is
visited whenever that is positive, and training starts from a level of one day of mean demand.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .forecasting import quantile_at, sigma_from_quantiles
from .network import SubsetRouting
from .policies import Context, Instance, cumulative_quantile, projected_waste, repair

torch.set_num_threads(1)


def static_features(X: np.ndarray, mu: np.ndarray, qs: np.ndarray, mean_store: np.ndarray, H: int) -> np.ndarray:
    """Per-store forecast features of the current day (rows k = 0 of the forecast matrix)."""
    N = mu.shape[0]
    sig = sigma_from_quantiles(qs[:, 0])
    s = mean_store
    return np.column_stack([X[np.arange(N) * H, 3:], mu[:, 0] / s, sig / s, (qs[:, 0, -1] - qs[:, 0, 0]) / s])


def stock_features(B: np.ndarray, mean_store: np.ndarray) -> np.ndarray:
    st = B / mean_store[:, None]
    return np.column_stack([st, st.sum(1)])


class RuleDeltaNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.double()

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class ULDFLModel:
    def __init__(self, lam: int, alpha: float, beta: float, mean_store: np.ndarray, n_in: int,
                 x_mean: np.ndarray, x_std: np.ndarray, hidden: int = 32, kind: str = "ul"):
        self.lam, self.alpha, self.beta, self.kind = lam, alpha, beta, kind
        self.mean_store = np.asarray(mean_store, dtype=float)
        self.x_mean, self.x_std, self.hidden = x_mean, x_std, hidden
        self.net = RuleDeltaNet(n_in, hidden)

    def state(self) -> dict:
        return {"lam": self.lam, "alpha": self.alpha, "beta": self.beta, "mean_store": self.mean_store,
                "n_in": len(self.x_mean), "x_mean": self.x_mean, "x_std": self.x_std, "hidden": self.hidden,
                "kind": self.kind,
                "weights": {k: v.detach().cpu().numpy().copy() for k, v in self.net.state_dict().items()}}

    @classmethod
    def from_state(cls, st: dict) -> "ULDFLModel":
        m = cls(st["lam"], st["alpha"], st["beta"], st["mean_store"], st["n_in"], st["x_mean"], st["x_std"],
                st["hidden"], st.get("kind", "ul"))
        m.net.load_state_dict({k: torch.tensor(v) for k, v in st["weights"].items()})
        m.net.eval()
        return m

    def delta(self, feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.from_numpy((feats - self.x_mean) / self.x_std)).numpy()


def rule_terms(mu, qs, lam, alpha, beta):
    """Order-up-to level, its forecast scale and the trigger threshold of UL_lambda."""
    sig = sigma_from_quantiles(qs)
    lam = min(lam, mu.shape[1])
    S = cumulative_quantile(mu, sig, qs, lam, alpha)
    scale = np.sqrt((sig[:, :lam] ** 2).sum(1))
    return S, scale, quantile_at(qs[:, 0], beta), lam


class ULDFLPolicy:
    """UL_lambda with a learned, context- and stock-age-dependent order-up-to correction."""

    def __init__(self, inst: Instance, router: SubsetRouting, model: ULDFLModel):
        self.inst, self.router, self.model = inst, router, model
        self.name = {"nn": "NN-DFL", "nb": "NB-DFL"}.get(model.kind, "UL-DFL")

    def decide(self, ctx: Context) -> tuple[np.ndarray, dict]:
        md = self.model
        H = ctx.mu.shape[1]
        feats = np.column_stack([static_features(ctx.X, ctx.mu, ctx.qs, md.mean_store, H),
                                 stock_features(ctx.B, md.mean_store)])
        if md.kind == "nn":
            q = md.mean_store * np.maximum(1.0 + md.delta(feats), 0.0)
            return repair(q, np.zeros_like(q), self.router, self.inst, ctx.B), {}
        if md.kind == "nb":
            S = md.mean_store * np.maximum(1.0 + md.delta(feats), 0.0)
            q = np.maximum(S - (ctx.B.sum(1) - projected_waste(ctx.B, ctx.mu, 1)), 0.0)
            return repair(q, np.zeros_like(q), self.router, self.inst, ctx.B), {}
        S, scale, thr, lam = rule_terms(ctx.mu, ctx.qs, md.lam, md.alpha, md.beta)
        S = S + scale * md.delta(feats)
        stock = ctx.B.sum(1)
        trigger = stock < thr
        usable = stock - projected_waste(ctx.B, ctx.mu, lam)
        q = np.where(trigger, np.maximum(S - usable, 0.0), 0.0)
        urgent = np.where(trigger, np.maximum(thr - stock, 0.0), 0.0)
        return repair(q, urgent, self.router, self.inst, ctx.B), {}


# ---------------------------------------------------------------------------- differentiable rollout
def _torch_projected_waste(B: torch.Tensor, mu: torch.Tensor, window: int) -> torch.Tensor:
    stock = B.clone()
    waste = torch.zeros(B.shape[:-1], dtype=B.dtype)
    for k in range(min(window, B.shape[-1])):
        rem = mu[..., min(k, mu.shape[-1] - 1)]
        cols = []
        for r in range(stock.shape[-1]):
            take = torch.minimum(stock[..., r], rem)
            cols.append(stock[..., r] - take)
            rem = rem - take
        waste = waste + cols[0]
        stock = torch.stack(cols[1:] + [torch.zeros_like(cols[0])], dim=-1)
    return waste


def _torch_step(B: torch.Tensor, q: torch.Tensor, d: torch.Tensor):
    avail = torch.cat([B, q[..., None]], dim=-1)
    rem = d
    cols = []
    for r in range(avail.shape[-1]):
        take = torch.minimum(avail[..., r], rem)
        cols.append(avail[..., r] - take)
        rem = rem - take
    waste = cols[0]
    Bn = torch.stack(cols[1:], dim=-1)
    return Bn, rem, waste, Bn.sum(-1)


def episode_tensors(D: dict, split: str, idx, start: int, n_days: int, H: int, md: ULDFLModel):
    """Stack the B-independent inputs of episodes idx: features, rule terms and demand."""
    F, S, SC, TH, MU, DEM = [], [], [], [], [], []
    for e in idx:
        ep, fc = D["episodes"][split][e], D["forecasts"][split][e]
        f_e, s_e, sc_e, th_e = [], [], [], []
        for t in range(n_days):
            s, sc, th, _ = rule_terms(fc["mu"][t], fc["qs"][t], md.lam, md.alpha, md.beta)
            f_e.append(static_features(fc["X"][t], fc["mu"][t], fc["qs"][t], md.mean_store, H))
            s_e.append(s), sc_e.append(sc), th_e.append(th)
        F.append(f_e), S.append(s_e), SC.append(sc_e), TH.append(th_e)
        MU.append(fc["mu"][:n_days])
        DEM.append(ep["demand"][:, start:start + n_days].T)
    return tuple(np.asarray(a, dtype=float) for a in (F, S, SC, TH, MU, DEM))  # (E, T, N, ...)


def rollout_cost(md: ULDFLModel, inst: Instance, tens, lam: int) -> torch.Tensor:
    F, S, SC, TH, MU, DEM = (torch.from_numpy(a) for a in tens)
    E, T, N = S.shape
    c = inst.costs
    cap = torch.as_tensor(inst.store_cap, dtype=torch.float64)
    fleet = float(inst.n_vehicles * inst.vehicle_cap)
    ms = torch.as_tensor(md.mean_store, dtype=torch.float64)
    xm, xs = torch.as_tensor(md.x_mean), torch.as_tensor(md.x_std)
    B = torch.zeros(E, N, inst.shelf_life - 1, dtype=torch.float64)
    total = torch.zeros(E, dtype=torch.float64)
    for t in range(T):
        st = (B / ms[:, None]).detach()
        x = torch.cat([F[:, t], st, st.sum(-1, keepdim=True)], dim=-1)
        stock = B.sum(-1)
        if md.kind == "nn":
            q = ms * torch.relu(1.0 + md.net((x - xm) / xs))
        elif md.kind == "nb":
            Sx = ms * torch.relu(1.0 + md.net((x - xm) / xs))
            q = torch.relu(Sx - (stock - _torch_projected_waste(B, MU[:, t], 1)))
        else:
            Sx = S[:, t] + SC[:, t] * md.net((x - xm) / xs)
            trig = (stock < TH[:, t]).to(B.dtype).detach()
            usable = stock - _torch_projected_waste(B, MU[:, t], lam)
            q = trig * torch.relu(Sx - usable)
        q = torch.minimum(q, torch.relu(cap - stock))
        q = q * torch.clamp(fleet / (q.sum(-1, keepdim=True) + 1e-9), max=1.0)
        B, lost, waste, carried = _torch_step(B, q, DEM[:, t])
        total = total + (c.p_lost * lost + c.p_waste * waste + c.h * carried).sum(-1)
    return total


def train_ul_dfl(inst: Instance, D: dict, spec: dict, val_fn, start: int, n_days: int, epochs: int = 30,
                 lr: float = 1e-3, batch: int = 10, l2: float = 1e-4, seed: int = 0, log=print, kind: str = "ul"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H = inst.horizon
    n_train = len(D["episodes"]["train"])
    probe = ULDFLModel(spec["lam"], spec["alpha"], spec["beta"], D["mean_store"], 1, np.zeros(1), np.ones(1))
    tens = episode_tensors(D, "train", range(n_train), start, n_days, H, probe)
    # feature normalisation: static part from the data, stock part from typical stock levels
    Fs = tens[0].reshape(-1, tens[0].shape[-1])
    m1 = inst.shelf_life - 1
    x_mean = np.concatenate([Fs.mean(0), np.full(m1, 0.5), [1.0]])
    x_std = np.concatenate([Fs.std(0) + 1e-6, np.full(m1, 0.5), [1.0]])
    md = ULDFLModel(spec["lam"], spec["alpha"], spec["beta"], D["mean_store"], len(x_mean), x_mean, x_std,
                    kind=kind)
    tag = {"nn": "NN-DFL", "nb": "NB-DFL"}.get(kind, "UL-DFL")
    lam = min(spec["lam"], H)
    opt = torch.optim.Adam(md.net.parameters(), lr=lr)
    scale = float(D["mean_store"].sum() * inst.costs.p_lost) * n_days / 10.0
    best_state = md.state()
    best_val = val_fn(best_state)
    history = [{"epoch": 0, "val_cost": best_val, "train_loss": np.nan}]
    start = {"nn": "one mean day per day", "nb": "up to one mean day"}.get(kind, "tuned UL")
    log(f"  {tag} epoch 0 ({start}) val={best_val:.2f}")
    for ep in range(1, epochs + 1):
        order = rng.permutation(n_train)
        losses = []
        md.net.train()
        for b0 in range(0, n_train, batch):
            idx = order[b0:b0 + batch]
            sub = tuple(a[idx] for a in tens)
            opt.zero_grad()
            loss = rollout_cost(md, inst, sub, lam).mean() / scale
            reg = l2 * sum((p ** 2).sum() for p in md.net.head.parameters())
            (loss + reg).backward()
            torch.nn.utils.clip_grad_norm_(md.net.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        md.net.eval()
        st = md.state()
        v = val_fn(st)
        history.append({"epoch": ep, "val_cost": v, "train_loss": float(np.mean(losses))})
        log(f"  {tag} epoch {ep} train={np.mean(losses):.4f} val={v:.2f}")
        if v < best_val:
            best_val, best_state = v, copy.deepcopy(st)
    return ULDFLModel.from_state(best_state), history
