"""Age-structured perishable inventory dynamics.

State convention. ``B`` has shape (N, m-1): ``B[i, r-1]`` is the number of units at
store i, at the start of a day and before delivery, that can still be sold on
``r`` days including today (r = 1..m-1). A fresh delivery can be sold on ``m``
days. Within a day: (1) the delivery arrives, (2) demand is served from the oldest
units first (FIFO; an optional fraction ``lifo_share`` of demand takes the freshest
units), (3) unmet demand is lost, (4) unsold units with one remaining day expire,
(5) the remaining units are carried overnight at holding cost h and their remaining
life falls by one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CostParams:
    h: float = 1.0        # holding cost per unit carried overnight
    p_waste: float = 5.0  # outdating (expiry) cost per unit
    p_lost: float = 20.0  # lost-sales penalty per unit
    c_route: float = 1.0  # cost per distance unit travelled


def step_fifo(B: np.ndarray, q: np.ndarray, d: np.ndarray, lifo_share: float = 0.0):
    """One day of FIFO dynamics for all stores. Returns (B_next, sold, lost, waste, carried).

    B: (N, m-1) state, q: (N,) deliveries, d: (N,) demand.
    """
    avail = np.concatenate([B, q[:, None]], axis=1).astype(float)  # columns r=1..m
    d_fifo = d * (1.0 - lifo_share)
    d_lifo = d * lifo_share
    rem = d_fifo.astype(float).copy()
    for r in range(avail.shape[1]):  # oldest first
        take = np.minimum(avail[:, r], rem)
        avail[:, r] -= take
        rem -= take
    rem_l = d_lifo.astype(float).copy()
    for r in range(avail.shape[1] - 1, -1, -1):  # freshest first
        take = np.minimum(avail[:, r], rem_l)
        avail[:, r] -= take
        rem_l -= take
    lost = rem + rem_l
    waste = avail[:, 0].copy()
    B_next = avail[:, 1:].copy()
    carried = B_next.sum(axis=1)
    sold = d - lost
    return B_next, sold, lost, waste, carried


def simulate_plan(B0: np.ndarray, Q: np.ndarray, D: np.ndarray, costs: CostParams, lifo_share: float = 0.0):
    """Simulate a fixed delivery plan Q (N, H) under demand D (N, H) from state B0.

    Returns a dict of per-day totals (inventory part only; routing is added by callers).
    """
    B = B0.copy()
    out = {"lost": [], "waste": [], "carried": [], "sold": [], "B": [B.copy()]}
    for t in range(Q.shape[1]):
        B, sold, lost, waste, carried = step_fifo(B, Q[:, t], D[:, t], lifo_share)
        out["lost"].append(lost)
        out["waste"].append(waste)
        out["carried"].append(carried)
        out["sold"].append(sold)
        out["B"].append(B.copy())
    for k in ("lost", "waste", "carried", "sold"):
        out[k] = np.stack(out[k], axis=1)
    out["inv_cost"] = (costs.h * out["carried"].sum() + costs.p_waste * out["waste"].sum()
                       + costs.p_lost * out["lost"].sum())
    return out


def torch_fifo_cost(B0: torch.Tensor, Q: torch.Tensor, D: torch.Tensor, costs: CostParams) -> torch.Tensor:
    """Differentiable (a.e.) FIFO inventory cost of delivery plans.

    B0: (..., N, m-1), Q and D: (..., N, H). Returns cost with shape (...).
    Autograd returns a subgradient of the piecewise-linear FIFO cost with respect to Q,
    which coincides with an optimal dual of the recourse LP (Proposition 2).
    """
    B = B0
    total = torch.zeros(Q.shape[:-2], dtype=Q.dtype, device=Q.device)
    for t in range(Q.shape[-1]):
        avail = torch.cat([B, Q[..., t:t + 1]], dim=-1)
        rem = D[..., t]
        cols = []
        for r in range(avail.shape[-1]):
            take = torch.minimum(avail[..., r], rem)
            cols.append(avail[..., r] - take)
            rem = rem - take
        lost = rem
        waste = cols[0]
        B = torch.stack(cols[1:], dim=-1) if len(cols) > 1 else avail[..., :0]
        total = total + (costs.p_lost * lost.sum(-1) + costs.p_waste * waste.sum(-1)
                         + costs.h * B.sum(dim=(-1, -2)))
    return total
