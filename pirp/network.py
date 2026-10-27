"""Network geometry, exact subset TSP costs, and exact capacitated route packing.

With at most 12 stores every subset of stores can be enumerated. ``SubsetRouting``
precomputes the exact travelling-salesman cost of visiting each subset from the
depot (Held-Karp dynamic programme) and uses it to solve, exactly, the capacitated
vehicle-routing problem for a given set of delivery quantities: choose at most K
disjoint routes, each with load at most Q, covering all stores with positive
delivery, at minimum total distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# Depot + first ten customers of Solomon R101, C101 and RC101 (time windows ignored).
SOLOMON_LAYOUTS: dict[str, dict] = {
    "R1": {
        "depot": (35.0, 35.0),
        "stores": [(41, 49), (35, 17), (55, 45), (55, 20), (15, 30),
                   (25, 30), (20, 50), (10, 43), (55, 60), (30, 60)],
    },
    "C1": {
        "depot": (40.0, 50.0),
        "stores": [(45, 68), (45, 70), (42, 66), (42, 68), (42, 65),
                   (40, 69), (40, 66), (38, 68), (38, 70), (35, 66)],
    },
    "RC1": {
        "depot": (40.0, 50.0),
        "stores": [(25, 85), (22, 75), (22, 85), (20, 80), (20, 85),
                   (18, 75), (15, 75), (15, 80), (30, 50), (35, 35)],
    },
}


def distance_matrix(depot: tuple[float, float], stores: list[tuple[float, float]]) -> np.ndarray:
    pts = np.array([depot, *stores], dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def held_karp_all_subsets(dist: np.ndarray) -> np.ndarray:
    """Exact TSP tour cost (depot -> subset -> depot) for every subset of stores.

    ``dist`` is (N+1)x(N+1) with the depot at index 0. Returns an array of length 2**N
    indexed by bitmask over stores 0..N-1 (store s is node s+1).
    """
    n = dist.shape[0] - 1
    full = 1 << n
    inf = np.inf
    # g[mask, j]: shortest path from depot visiting exactly `mask`, ending at store j (j in mask)
    g = np.full((full, n), inf)
    for j in range(n):
        g[1 << j, j] = dist[0, j + 1]
    store_d = dist[1:, 1:]
    for mask in range(1, full):
        row = g[mask]
        if not np.isfinite(row).any():
            continue
        for j in range(n):
            if not (mask >> j) & 1 or not np.isfinite(row[j]):
                continue
            base = row[j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                val = base + store_d[j, k]
                if val < g[nm, k]:
                    g[nm, k] = val
    tour = np.zeros(full)
    for mask in range(1, full):
        tour[mask] = np.min(g[mask] + dist[1:, 0])
    return tour


@dataclass
class SubsetRouting:
    """Exact routing oracle for small networks via subset enumeration."""

    dist: np.ndarray
    n_vehicles: int
    capacity: float
    tour_cost: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.n = self.dist.shape[0] - 1
        if self.n > 12:
            raise ValueError("Subset enumeration is limited to at most 12 stores")
        self.tour_cost = held_karp_all_subsets(self.dist)
        self._bits = np.array([[(m >> i) & 1 for i in range(self.n)] for m in range(1 << self.n)], dtype=bool)

    @staticmethod
    def mask_of(stores) -> int:
        m = 0
        for s in stores:
            m |= 1 << int(s)
        return m

    def route_cost(self, mask: int) -> float:
        return float(self.tour_cost[mask])

    def pack(self, q: np.ndarray, tol: float = 1e-6) -> tuple[float, list[int]] | None:
        """Exact min-distance packing of positive deliveries into <= K routes of load <= Q.

        Returns (cost, route_masks) or None when no feasible packing exists.
        """
        visit = [i for i in range(self.n) if q[i] > tol]
        if not visit:
            return 0.0, []
        total = float(q[visit].sum())
        if total > self.n_vehicles * self.capacity + tol:
            return None
        vmask = self.mask_of(visit)
        return _best_partition(vmask, tuple(float(x) for x in q), self.n_vehicles, self.capacity + tol,
                               self._tour_tuple())

    def _tour_tuple(self):
        if not hasattr(self, "_tt"):
            self._tt = tuple(float(x) for x in self.tour_cost)
        return self._tt


@lru_cache(maxsize=200_000)
def _best_partition(vmask: int, q: tuple, k: int, cap: float, tour: tuple):
    """Min-cost partition of `vmask` into at most k subsets each with load <= cap."""
    if vmask == 0:
        return 0.0, []
    load = sum(q[i] for i in range(len(q)) if (vmask >> i) & 1)
    if k == 1:
        return (tour[vmask], [vmask]) if load <= cap else None
    if load <= cap:
        best = (tour[vmask], [vmask])
    else:
        best = None
    # the subset containing the lowest set bit is enumerated explicitly
    low = vmask & -vmask
    rest = vmask ^ low
    sub = rest
    while True:
        first = sub | low
        l1 = sum(q[i] for i in range(len(q)) if (first >> i) & 1)
        if l1 <= cap and first != vmask:
            other = _best_partition(vmask ^ first, q, k - 1, cap, tour)
            if other is not None:
                c = tour[first] + other[0]
                if best is None or c < best[0] - 1e-12:
                    best = (c, [first, *other[1]])
        if sub == 0:
            break
        sub = (sub - 1) & rest
    return best


def build_network(layout: str = "R1", n_stores: int = 10) -> tuple[np.ndarray, np.ndarray]:
    spec = SOLOMON_LAYOUTS[layout]
    stores = spec["stores"][:n_stores]
    coords = np.array([spec["depot"], *stores], dtype=float)
    return coords, distance_matrix(spec["depot"], stores)
