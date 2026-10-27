import itertools

import numpy as np
import pytest
import torch

from pirp.forecasting import QUANTILE_LEVELS, quantile_at
from pirp.milp import PIRPModel
from pirp.network import SubsetRouting, build_network, held_karp_all_subsets
from pirp.policies import robust_targets, sample_scenarios
from pirp.simulator import CostParams, simulate_plan, step_fifo, torch_fifo_cost


def test_fifo_step_oldest_first_and_expiry():
    B = np.array([[3.0, 5.0]])           # 3 units expire today, 5 units have two days left
    B2, sold, lost, waste, carried = step_fifo(B, np.array([10.0]), np.array([6.0]))
    assert sold[0] == 6 and lost[0] == 0
    assert waste[0] == 0                 # the 3 oldest were sold first
    assert np.allclose(B2, [[2.0, 10.0]])
    B3, sold, lost, waste, carried = step_fifo(B2, np.array([0.0]), np.array([1.0]))
    assert waste[0] == 1.0 and np.allclose(B3, [[10.0, 0.0]])


def test_lifo_share_takes_freshest():
    B = np.array([[4.0, 0.0]])
    _, _, _, waste, _ = step_fifo(B, np.array([4.0]), np.array([4.0]), lifo_share=1.0)
    assert waste[0] == 4.0


def test_torch_fifo_matches_numpy():
    rng = np.random.default_rng(0)
    costs = CostParams()
    for _ in range(50):
        B0 = rng.integers(0, 10, (4, 2)).astype(float)
        Q = rng.integers(0, 20, (4, 3)).astype(float)
        D = rng.integers(0, 25, (4, 3)).astype(float)
        a = simulate_plan(B0, Q, D, costs)["inv_cost"]
        b = float(torch_fifo_cost(torch.tensor(B0), torch.tensor(Q), torch.tensor(D), costs))
        assert abs(a - b) < 1e-9


def _brute_tsp(dist, stores):
    best = 0.0 if not stores else np.inf
    for perm in itertools.permutations(stores):
        path = [0, *[s + 1 for s in perm], 0]
        best = min(best, sum(dist[path[i], path[i + 1]] for i in range(len(path) - 1)))
    return best


def test_held_karp_exact_small():
    _, dist = build_network("R1", 6)
    tour = held_karp_all_subsets(dist)
    for mask in (1, 5, 13, 63):
        stores = [i for i in range(6) if (mask >> i) & 1]
        assert abs(tour[mask] - _brute_tsp(dist, stores)) < 1e-9


def test_pack_exact_against_enumeration():
    _, dist = build_network("R1", 7)
    r = SubsetRouting(dist, 2, 30.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.integers(0, 12, 7).astype(float)
        res = r.pack(q)
        visit = [i for i in range(7) if q[i] > 0]
        best = np.inf
        for assign in itertools.product([0, 1], repeat=len(visit)):
            g = [[v for v, a in zip(visit, assign) if a == j] for j in (0, 1)]
            if all(q[gg].sum() <= 30.0 for gg in g):
                best = min(best, sum(_brute_tsp(dist, gg) for gg in g))
        if np.isinf(best):
            assert res is None
        else:
            assert abs(res[0] - best) < 1e-9


@pytest.mark.parametrize("S", [1, 3])
def test_lookahead_milp_feasible_and_respects_capacity(S):
    _, dist = build_network("R1")
    cap = np.full(10, 40.0)
    mdl = PIRPModel(dist, cap, 2, 90.0, 3, 2, CostParams(), n_scen=S)
    rng = np.random.default_rng(2)
    B0 = rng.integers(0, 10, (10, 2)).astype(float)
    d = rng.gamma(5, 4, (S, 10, 2)) if S > 1 else rng.gamma(5, 4, (10, 2))
    res = mdl.solve(B0, d, time_limit=60, mip_rel_gap=0.005)
    assert res.success
    assert np.all(res.q[:, 0] + B0.sum(1) <= cap + 1e-6)
    router = SubsetRouting(dist, 2, 90.0)
    assert router.pack(res.q[:, 0]) is not None


def test_quantile_interpolation_and_scenarios():
    qs = np.tile(np.linspace(1, 11, len(QUANTILE_LEVELS)), (2, 3, 1))
    assert np.allclose(quantile_at(qs, 0.5), qs[..., 5])
    sc = sample_scenarios(qs, 20, 0)
    assert sc.min() >= qs[..., 0].min() - 1e-9 and sc.max() <= qs[..., -1].max() + 1e-9


def test_robust_targets_reduce_to_mean_at_zero_budget():
    rng = np.random.default_rng(3)
    mu = rng.uniform(5, 20, (4, 3))
    qs = np.sort(rng.uniform(0, 40, (4, 3, len(QUANTILE_LEVELS))), axis=-1)
    assert np.allclose(robust_targets(mu, qs, 0.0), mu)
    assert np.all(robust_targets(mu, qs, 1.0).sum(1) >= mu.sum(1) - 1e-9)
