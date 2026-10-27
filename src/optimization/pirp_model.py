"""
Multi-Period Perishable Inventory-Routing Problem (PIRP) Mathematical Model.

Formulates the PIRP using HiGHS via scipy.optimize (milp and linprog).
Supports:
  1. Discrete routing with an explicit choice between MILP and a checked heuristic
  2. Continuous LP relaxation mode (x in [0, 1])
  3. FIFO perishable inventory age tracking (tau in {2, 3, 5})
  4. Complete recourse (emergency stockout y_pos, spoilage disposal y_neg)
  5. Cost breakdown: C_route, C_hold, C_spoil, C_stock, C_total.

Discrete solves use a constructive heuristic by default. Set ``use_milp=True``
to invoke HiGHS branch-and-cut; check the returned solver diagnostics before
claiming optimality.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix


@dataclass
class PIRPSolution:
    """Stores the complete optimization results from PIRPModel."""
    success: bool
    status: int
    message: str
    objective_val: float
    routing_cost: float
    holding_cost: float
    spoilage_cost: float
    stockout_cost: float
    total_cost: float
    # Decision arrays
    x_routes: np.ndarray       # Shape (N+1, N+1, K, T)
    q_deliveries: np.ndarray   # Shape (N, K, T)
    q_total: np.ndarray        # Shape (N, T)
    I_inventory: np.ndarray    # Shape (N, T, tau-1)
    w_consumed: np.ndarray     # Shape (N, T, tau)
    y_pos: np.ndarray          # Shape (N, T)
    y_neg: np.ndarray          # Shape (N, T)
    solve_time_sec: float
    is_lp_relaxation: bool
    solver_name: str = ""
    mip_gap: float | None = None
    mip_dual_bound: float | None = None
    mip_node_count: int | None = None
    is_optimal: bool = False
    used_heuristic_fallback: bool = False


class PIRPModel:
    """
    Parametric optimization model for the Multi-Period Perishable Inventory-Routing Problem.
    Constructs the sparse constraint matrix and solves via HiGHS.
    """

    def __init__(
        self,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        aging_surcharge_eps: float = 0.01,
        spoilage_penalty_ps: float = 5.0,
        stockout_penalty_pu: float = 20.0,
        distance_matrix: np.ndarray | None = None,
        continuous_relaxation: bool = False,
        initial_inventory: np.ndarray | None = None,
    ):
        self.N = num_stores
        self.T = horizon_T
        self.K = fleet_size_K
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = shelf_life_tau
        self.h = float(holding_cost_h)
        self.aging_surcharge_eps = float(aging_surcharge_eps)
        self.ps = float(spoilage_penalty_ps)
        self.pu = float(stockout_penalty_pu)
        self.continuous_relaxation = continuous_relaxation
        self.solver_diagnostics: list[dict[str, object]] = []
        if initial_inventory is None:
            self.initial_inventory = np.zeros((self.N, self.tau - 1), dtype=np.float64)
        else:
            self.initial_inventory = np.asarray(initial_inventory, dtype=np.float64)
            if self.initial_inventory.shape != (self.N, self.tau - 1):
                raise ValueError(
                    f"initial_inventory must have shape ({self.N}, {self.tau - 1}), "
                    f"got {self.initial_inventory.shape}"
                )
            if np.any(self.initial_inventory < 0) or np.any(~np.isfinite(self.initial_inventory)):
                raise ValueError("initial_inventory must contain finite nonnegative quantities")
            if np.any(self.initial_inventory.sum(axis=1) > self.Ci + 1e-8):
                raise ValueError("initial inventory exceeds store capacity")

        # Setup distance matrix (N+1 x N+1)
        if distance_matrix is not None:
            self.dist = np.array(distance_matrix, dtype=np.float64)
            assert self.dist.shape == (self.N + 1, self.N + 1), f"Expected shape ({self.N+1}, {self.N+1}), got {self.dist.shape}"
        else:
            # Default metric coordinates
            np.random.seed(42)
            coords = np.random.uniform(0, 100, size=(self.N + 1, 2))
            diff = coords[:, None, :] - coords[None, :, :]
            self.dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        self._build_variable_index_map()
        self._build_static_constraints()

    def _build_variable_index_map(self):
        """Indexes every decision variable in the flat optimization vector z."""
        self.var_map = {}
        idx = 0

        # 1. Routing x[i, j, k, t] for i != j in 0..N, k in 0..K-1, t in 0..T-1
        self.x_indices = {}
        for t in range(self.T):
            for k in range(self.K):
                for i in range(self.N + 1):
                    for j in range(self.N + 1):
                        if i != j:
                            self.x_indices[(i, j, k, t)] = idx
                            idx += 1
        self.num_x = len(self.x_indices)

        # 2. MTZ subtour auxiliary u[i, k, t] for i in 1..N, k in 0..K-1, t in 0..T-1
        self.u_indices = {}
        for t in range(self.T):
            for k in range(self.K):
                for i in range(1, self.N + 1):
                    self.u_indices[(i, k, t)] = idx
                    idx += 1
        self.num_u = len(self.u_indices)

        # 3. Deliveries q[i, k, t] for i in 1..N, k in 0..K-1, t in 0..T-1
        self.q_indices = {}
        for t in range(self.T):
            for k in range(self.K):
                for i in range(1, self.N + 1):
                    self.q_indices[(i, k, t)] = idx
                    idx += 1
        self.num_q = len(self.q_indices)

        # 4. Age-tracked inventory I[i, t, a] for i in 1..N, t in 0..T-1, a in 1..tau-1
        self.I_indices = {}
        for t in range(self.T):
            for i in range(1, self.N + 1):
                for a in range(1, self.tau):
                    self.I_indices[(i, t, a)] = idx
                    idx += 1
        self.num_I = len(self.I_indices)

        # 5. Consumption w[i, t, a] for i in 1..N, t in 0..T-1, a in 1..tau
        self.w_indices = {}
        for t in range(self.T):
            for i in range(1, self.N + 1):
                for a in range(1, self.tau + 1):
                    self.w_indices[(i, t, a)] = idx
                    idx += 1
        self.num_w = len(self.w_indices)

        # 6. Recourse y_pos[i, t] (stockout/emergency) for i in 1..N, t in 0..T-1
        self.y_pos_indices = {}
        for t in range(self.T):
            for i in range(1, self.N + 1):
                self.y_pos_indices[(i, t)] = idx
                idx += 1
        self.num_y_pos = len(self.y_pos_indices)

        # 7. Recourse y_neg[i, t] (spoilage/waste) for i in 1..N, t in 0..T-1
        self.y_neg_indices = {}
        for t in range(self.T):
            for i in range(1, self.N + 1):
                self.y_neg_indices[(i, t)] = idx
                idx += 1
        self.num_y_neg = len(self.y_neg_indices)

        self.total_vars = idx

        # Default objective cost vector
        self.base_c = np.zeros(self.total_vars, dtype=np.float64)
        for (i, j, k, t), x_idx in self.x_indices.items():
            self.base_c[x_idx] = self.dist[i, j]
        for (i, t, a), I_idx in self.I_indices.items():
            self.base_c[I_idx] = self.h + (a - 1) * self.aging_surcharge_eps
        for (i, t), yp_idx in self.y_pos_indices.items():
            self.base_c[yp_idx] = self.pu
        for (i, t), yn_idx in self.y_neg_indices.items():
            self.base_c[yn_idx] = self.ps

        # Variable lower and upper bounds
        self.lb = np.zeros(self.total_vars, dtype=np.float64)
        self.ub = np.full(self.total_vars, np.inf, dtype=np.float64)

        # x in [0, 1]
        for x_idx in self.x_indices.values():
            self.ub[x_idx] = 1.0

        # u in [1, N]
        for u_idx in self.u_indices.values():
            self.lb[u_idx] = 1.0
            self.ub[u_idx] = float(self.N)

        # q in [0, min(Q, Ci)]
        for q_idx in self.q_indices.values():
            self.ub[q_idx] = min(self.Q, self.Ci)

        # I in [0, Ci]
        for I_idx in self.I_indices.values():
            self.ub[I_idx] = self.Ci

        # Integrality vector: 1 for integer (x), 0 for continuous
        self._x_idx_list = np.array(list(self.x_indices.values()), dtype=np.intp)
        self.integrality = np.zeros(self.total_vars, dtype=np.int32)
        if not self.continuous_relaxation:
            self.integrality[self._x_idx_list] = 1

        # Precompute structured index arrays for fast extraction in solve()
        self._x_i = np.array([k[0] for k in self.x_indices], dtype=np.intp)
        self._x_j = np.array([k[1] for k in self.x_indices], dtype=np.intp)
        self._x_k = np.array([k[2] for k in self.x_indices], dtype=np.intp)
        self._x_t = np.array([k[3] for k in self.x_indices], dtype=np.intp)

        self._q_idx_arr = np.empty((self.N, self.K, self.T), dtype=np.intp)
        for (i, k, t), q_idx in self.q_indices.items():
            self._q_idx_arr[i - 1, k, t] = q_idx

        self._I_idx_arr = np.empty((self.N, self.T, self.tau - 1), dtype=np.intp)
        for (i, t, a), I_idx in self.I_indices.items():
            self._I_idx_arr[i - 1, t, a - 1] = I_idx

        self._w_idx_arr = np.empty((self.N, self.T, self.tau), dtype=np.intp)
        for (i, t, a), w_idx in self.w_indices.items():
            self._w_idx_arr[i - 1, t, a - 1] = w_idx

        self._yp_idx_arr = np.empty((self.N, self.T), dtype=np.intp)
        for (i, t), yp_idx in self.y_pos_indices.items():
            self._yp_idx_arr[i - 1, t] = yp_idx

        self._yn_idx_arr = np.empty((self.N, self.T), dtype=np.intp)
        for (i, t), yn_idx in self.y_neg_indices.items():
            self._yn_idx_arr[i - 1, t] = yn_idx

    def set_continuous_relaxation(self, continuous: bool):
        """Updates continuous relaxation mode in-place without rebuilding static structures."""
        self.continuous_relaxation = continuous
        self.integrality.fill(0)
        if not continuous:
            self.integrality[self._x_idx_list] = 1

    def _build_static_constraints(self):
        """
        Builds static constraints independent of demand:
          - Vehicle depot departure and flow conservation
          - Delivery visit linking and vehicle payload capacity
          - MTZ subtour elimination
          - FIFO age inventory balance (fresh, intermediate, spoilage)
          - Store physical storage capacity
        """
        # Equality constraints: A_eq * z = b_eq
        eq_rows = []
        eq_cols = []
        eq_vals = []
        eq_rhs = []
        self._initial_rhs_indices: dict[tuple[int, int], int] = {}
        eq_count = 0

        # Inequality constraints: A_ub * z <= b_ub
        ub_rows = []
        ub_cols = []
        ub_vals = []
        ub_ub = []
        ub_count = 0

        # --- 1. Vehicle Flow Conservation (Equality) ---
        for t in range(self.T):
            for k in range(self.K):
                # Depot flow balance: sum_j x[0, j] - sum_i x[i, 0] = 0
                for j in range(1, self.N + 1):
                    eq_rows.append(eq_count)
                    eq_cols.append(self.x_indices[(0, j, k, t)])
                    eq_vals.append(1.0)
                for i in range(1, self.N + 1):
                    eq_rows.append(eq_count)
                    eq_cols.append(self.x_indices[(i, 0, k, t)])
                    eq_vals.append(-1.0)
                eq_rhs.append(0.0)
                eq_count += 1

                # Retail store flow conservation: sum_{j!=i} x[j, i] - sum_{j!=i} x[i, j] = 0
                for i in range(1, self.N + 1):
                    for j in range(self.N + 1):
                        if j != i:
                            eq_rows.append(eq_count)
                            eq_cols.append(self.x_indices[(j, i, k, t)])
                            eq_vals.append(1.0)

                            eq_rows.append(eq_count)
                            eq_cols.append(self.x_indices[(i, j, k, t)])
                            eq_vals.append(-1.0)
                    eq_rhs.append(0.0)
                    eq_count += 1

        # --- 2. Vehicle Depot Departure & Capacity (Inequality) ---
        for t in range(self.T):
            for k in range(self.K):
                # At most 1 departure: sum_j x[0, j, k, t] <= 1
                for j in range(1, self.N + 1):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.x_indices[(0, j, k, t)])
                    ub_vals.append(1.0)
                ub_ub.append(1.0)
                ub_count += 1

                # Vehicle capacity: sum_i q[i, k, t] - Q * sum_j x[0, j, k, t] <= 0
                for i in range(1, self.N + 1):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.q_indices[(i, k, t)])
                    ub_vals.append(1.0)
                for j in range(1, self.N + 1):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.x_indices[(0, j, k, t)])
                    ub_vals.append(-self.Q)
                ub_ub.append(0.0)
                ub_count += 1

            # Vehicle symmetry breaking: vehicle 0 used before vehicle 1
            if self.K >= 2:
                # sum_j x[0, j, 1, t] - sum_j x[0, j, 0, t] <= 0
                for j in range(1, self.N + 1):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.x_indices[(0, j, 1, t)])
                    ub_vals.append(1.0)
                    ub_rows.append(ub_count)
                    ub_cols.append(self.x_indices[(0, j, 0, t)])
                    ub_vals.append(-1.0)
                ub_ub.append(0.0)
                ub_count += 1

            # Store visit at most once per period: sum_k sum_{j!=i} x[j, i, k, t] <= 1
            for i in range(1, self.N + 1):
                for k in range(self.K):
                    for j in range(self.N + 1):
                        if j != i:
                            ub_rows.append(ub_count)
                            ub_cols.append(self.x_indices[(j, i, k, t)])
                            ub_vals.append(1.0)
                ub_ub.append(1.0)
                ub_count += 1

        # --- 3. Delivery Visit Linking (Inequality) ---
        # q[i, k, t] <= min(Q, Ci) * sum_{j!=i} x[j, i, k, t]
        M_link = min(self.Q, self.Ci)
        for t in range(self.T):
            for k in range(self.K):
                for i in range(1, self.N + 1):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.q_indices[(i, k, t)])
                    ub_vals.append(1.0)
                    for j in range(self.N + 1):
                        if j != i:
                            ub_rows.append(ub_count)
                            ub_cols.append(self.x_indices[(j, i, k, t)])
                            ub_vals.append(-M_link)
                    ub_ub.append(0.0)
                    ub_count += 1

                # Linking store visit to depot departure: sum_{j!=i} x[j, i, k, t] - sum_j x[0, j, k, t] <= 0
                for i in range(1, self.N + 1):
                    for j in range(self.N + 1):
                        if j != i:
                            ub_rows.append(ub_count)
                            ub_cols.append(self.x_indices[(j, i, k, t)])
                            ub_vals.append(1.0)
                    for j in range(1, self.N + 1):
                        ub_rows.append(ub_count)
                        ub_cols.append(self.x_indices[(0, j, k, t)])
                        ub_vals.append(-1.0)
                    ub_ub.append(0.0)
                    ub_count += 1

        # --- 4. MTZ Subtour Elimination (Inequality) ---
        # u[i, k, t] - u[j, k, t] + N * x[i, j, k, t] <= N - 1  for i, j in 1..N, i != j
        for t in range(self.T):
            for k in range(self.K):
                for i in range(1, self.N + 1):
                    for j in range(1, self.N + 1):
                        if i != j:
                            ub_rows.append(ub_count)
                            ub_cols.append(self.u_indices[(i, k, t)])
                            ub_vals.append(1.0)

                            ub_rows.append(ub_count)
                            ub_cols.append(self.u_indices[(j, k, t)])
                            ub_vals.append(-1.0)

                            ub_rows.append(ub_count)
                            ub_cols.append(self.x_indices[(i, j, k, t)])
                            ub_vals.append(float(self.N))

                            ub_ub.append(float(self.N - 1))
                            ub_count += 1

        # --- 5. Perishable Inventory Dynamics (FIFO Flow) ---
        for t in range(self.T):
            for i in range(1, self.N + 1):
                # (a) Fresh Inventory Balance (a = 1):
                # I[i, t, 1] + w[i, t, 1] - sum_k q[i, k, t] = 0
                eq_rows.append(eq_count)
                eq_cols.append(self.I_indices[(i, t, 1)])
                eq_vals.append(1.0)

                eq_rows.append(eq_count)
                eq_cols.append(self.w_indices[(i, t, 1)])
                eq_vals.append(1.0)

                for k in range(self.K):
                    eq_rows.append(eq_count)
                    eq_cols.append(self.q_indices[(i, k, t)])
                    eq_vals.append(-1.0)
                eq_rhs.append(0.0)
                eq_count += 1

                # (b) Intermediate Age Balance (2 <= a <= tau-1):
                # If t == 0: I[i, 0, a] + w[i, 0, a] = I_init[i, a-1] (default 0)
                # If t > 0:  I[i, t, a] + w[i, t, a] - I[i, t-1, a-1] = 0
                for a in range(2, self.tau):
                    if t == 0:
                        self._initial_rhs_indices[(i - 1, a - 2)] = eq_count
                    eq_rows.append(eq_count)
                    eq_cols.append(self.I_indices[(i, t, a)])
                    eq_vals.append(1.0)

                    eq_rows.append(eq_count)
                    eq_cols.append(self.w_indices[(i, t, a)])
                    eq_vals.append(1.0)

                    if t > 0:
                        eq_rows.append(eq_count)
                        eq_cols.append(self.I_indices[(i, t - 1, a - 1)])
                        eq_vals.append(-1.0)
                        eq_rhs.append(0.0)
                    else:
                        eq_rhs.append(0.0)  # Assuming zero initial inventory
                    eq_count += 1

                # (c) Oldest Inventory & Spoilage Recourse (a = tau):
                # If t == 0: y_neg[i, 0] + w[i, 0, tau] = I_init[i, tau-1] (default 0)
                # If t > 0:  y_neg[i, t] + w[i, t, tau] - I[i, t-1, tau-1] = 0
                if t == 0:
                    self._initial_rhs_indices[(i - 1, self.tau - 2)] = eq_count
                eq_rows.append(eq_count)
                eq_cols.append(self.y_neg_indices[(i, t)])
                eq_vals.append(1.0)

                eq_rows.append(eq_count)
                eq_cols.append(self.w_indices[(i, t, self.tau)])
                eq_vals.append(1.0)

                if t > 0:
                    eq_rows.append(eq_count)
                    eq_cols.append(self.I_indices[(i, t - 1, self.tau - 1)])
                    eq_vals.append(-1.0)
                    eq_rhs.append(0.0)
                else:
                    eq_rhs.append(0.0)
                eq_count += 1

                # (d) Store Storage Capacity (Inequality):
                # sum_{a=1}^{tau-1} I[i, t, a] <= Ci
                for a in range(1, self.tau):
                    ub_rows.append(ub_count)
                    ub_cols.append(self.I_indices[(i, t, a)])
                    ub_vals.append(1.0)
                ub_ub.append(self.Ci)
                ub_count += 1

        self.static_eq_rows = eq_rows
        self.static_eq_cols = eq_cols
        self.static_eq_vals = eq_vals
        self.static_eq_rhs = eq_rhs
        self.static_eq_count = eq_count

        self.static_ub_rows = ub_rows
        self.static_ub_cols = ub_cols
        self.static_ub_vals = ub_vals
        self.static_ub_ub = ub_ub
        self.static_ub_count = ub_count

        # Pre-build complete constraint matrix structure (since LHS coefficients of demand rows are also static)
        all_eq_rows = list(self.static_eq_rows)
        all_eq_cols = list(self.static_eq_cols)
        all_eq_vals = list(self.static_eq_vals)
        all_eq_rhs_base = list(self.static_eq_rhs)
        cur_eq = self.static_eq_count

        for t in range(self.T):
            for i in range(1, self.N + 1):
                for a in range(1, self.tau + 1):
                    all_eq_rows.append(cur_eq)
                    all_eq_cols.append(self.w_indices[(i, t, a)])
                    all_eq_vals.append(1.0)
                all_eq_rows.append(cur_eq)
                all_eq_cols.append(self.y_pos_indices[(i, t)])
                all_eq_vals.append(1.0)
                all_eq_rhs_base.append(0.0)
                cur_eq += 1

        self._total_eq = cur_eq
        self._total_ub = self.static_ub_count
        self._total_constrs = self._total_eq + self._total_ub

        combined_rows = all_eq_rows + [r + self._total_eq for r in self.static_ub_rows]
        combined_cols = all_eq_cols + self.static_ub_cols
        combined_vals = all_eq_vals + self.static_ub_vals

        self._A_sparse = csc_matrix(
            (combined_vals, (combined_rows, combined_cols)),
            shape=(self._total_constrs, self.total_vars),
        )
        self._A_eq = self._A_sparse[: self._total_eq, :]
        self._A_ub = self._A_sparse[self._total_eq :, :]
        self._b_ub = np.array(self.static_ub_ub, dtype=np.float64)
        self._b_eq_template = np.array(all_eq_rhs_base, dtype=np.float64)
        self._lp_bounds = list(zip(self.lb, self.ub))
        self._milp_bounds = Bounds(self.lb, self.ub)

    def build_system(self, demand: np.ndarray) -> tuple[csc_matrix, np.ndarray, np.ndarray, int]:
        """
        Builds the complete linear constraint system [A_eq; A_ub] for a specific demand matrix.

        Args:
            demand: Array of shape (N, T) representing retailer demands d[i, t].

        Returns:
            A_combined: Sparse CSC matrix of constraints.
            b_l: Lower bounds for constraints.
            b_u: Upper bounds for constraints.
            total_eq: Number of equality constraints.
        """
        demand = np.asarray(demand, dtype=np.float64)
        assert demand.shape == (self.N, self.T), f"Expected demand shape ({self.N}, {self.T}), got {demand.shape}"

        b_l = np.empty(self._total_constrs, dtype=np.float64)
        b_u = np.empty(self._total_constrs, dtype=np.float64)

        static_eq_rhs = np.array(self._b_eq_template[: self.static_eq_count], copy=True)
        for (store_idx, age_idx), row_idx in self._initial_rhs_indices.items():
            static_eq_rhs[row_idx] = self.initial_inventory[store_idx, age_idx]
        b_l[: self.static_eq_count] = static_eq_rhs
        # Demand rows are ordered by t in 0..T-1, then i in 1..N -> demand.T.ravel()
        demand_flat = demand.T.ravel()
        b_l[self.static_eq_count : self._total_eq] = demand_flat
        b_u[: self._total_eq] = b_l[: self._total_eq]

        b_l[self._total_eq :] = -np.inf
        b_u[self._total_eq :] = self._b_ub

        return self._A_sparse, b_l, b_u, self._total_eq

    def _construct_integer_solution(
        self,
        demand: np.ndarray,
        c: np.ndarray,
        z_lp: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Constructs a guaranteed constraint-feasible integer solution z_int (x in {0, 1})
        via LP-guided replenishment, unsplittable vehicle bin-packing, 2-opt TSP routing,
        and exact FIFO perishable inventory dynamics.
        """
        if z_lp is None:
            _, _, b_u, num_eq = self.build_system(demand)
            res_lp = linprog(
                c=c,
                A_ub=self._A_ub,
                b_ub=self._b_ub,
                A_eq=self._A_eq,
                b_eq=b_u[:num_eq],
                bounds=self._lp_bounds,
                method="highs",
                options={"disp": False},
            )
            z_lp = res_lp.x if res_lp.success else np.zeros(self.total_vars, dtype=np.float64)

        q_lp_deliv = z_lp[self._q_idx_arr]  # (N, K, T)
        q_lp_tot = np.clip(np.sum(q_lp_deliv, axis=1), 0.0, min(self.Q, self.Ci))  # (N, T)

        # Multi-period route consolidation for distant stores over shelf-life window tau:
        # Consolidates deliveries [t0 .. t0+tau-1] into period t0 (up to store capacity Ci)
        # using spare period-t0 fleet capacity and capacity-neutral swaps of pre-positioned carryover
        # with periods t0+1 .. t0+tau-1, saving (tau - 1) distant vehicle routing visits.
        if self.tau <= self.T:
            distant_stores = sorted(
                range(self.N),
                key=lambda idx: (float(np.sum(demand[idx, : self.tau])) - 0.4 * self.dist[0, idx + 1]),
            )
            t0 = 0
            for i_idx in distant_stores[:2]:
                target_batch = min(min(self.Q, self.Ci), float(np.sum(demand[i_idx, t0 : t0 + self.tau])))
                future_moved = max(0.0, target_batch - float(q_lp_tot[i_idx, t0]))
                if future_moved <= 2.0:
                    continue
                period_load = float(np.sum(q_lp_tot[:, t0]))
                spare_t0 = max(0.0, self.K * self.Q - period_load)
                base_need = np.minimum(demand[:, t0], q_lp_tot[:, t0])
                swappable_by_j = np.maximum(0.0, q_lp_tot[:, t0] - 0.88 * base_need)
                swappable_by_j[i_idx] = 0.0
                total_swappable = float(np.sum(swappable_by_j))
                if future_moved <= spare_t0 + total_swappable + 1e-6:
                    need_swap = max(0.0, future_moved - spare_t0)
                    rem_to_cut = float(np.sum(q_lp_tot[i_idx, t0 + 1 : t0 + self.tau]))
                    q_lp_tot[i_idx, t0 + 1 : t0 + self.tau] = 0.0
                    if need_swap > 1e-7 and total_swappable > 1e-7:
                        weights_j = swappable_by_j / total_swappable
                        delta_j = need_swap * weights_j
                        q_lp_tot[:, t0] -= delta_j
                        # Restore deferred quantity across t0+1 .. t0+tau-1 where capacity was freed
                        if rem_to_cut > 1e-7:
                            for s in range(t0 + 1, t0 + self.tau):
                                share_s = min(need_swap, max(0.0, self.K * self.Q - float(np.sum(q_lp_tot[:, s]))))
                                if share_s > 1e-7:
                                    q_lp_tot[:, s] = np.minimum(
                                        min(self.Q, self.Ci), q_lp_tot[:, s] + share_s * weights_j
                                    )
                                    need_swap -= share_s
                                    if need_swap <= 1e-7:
                                        break
                    q_lp_tot[i_idx, t0] = target_batch

        z_int = np.zeros(self.total_vars, dtype=np.float64)
        # Initialize u variables to 1.0 (within [1, N])
        for u_idx in self.u_indices.values():
            z_int[u_idx] = 1.0

        q_int_tot = np.zeros((self.N, self.T), dtype=np.float64)

        for t in range(self.T):
            q_t = q_lp_tot[:, t].copy()
            active_stores = [i for i in range(1, self.N + 1) if q_t[i - 1] > 1e-6]
            if not active_stores:
                continue

            # Partition active stores across K vehicles (each store visited at most once)
            # For K=2 and N<=12, find the exact subset partition that maximizes total delivered load <= K*Q
            # while minimizing spatial cluster spread.
            m_act = len(active_stores)
            if self.K == 2 and m_act <= 12 and np.sum(q_t) > self.Q + 1e-6:
                loads = np.array([q_t[i - 1] for i in active_stores], dtype=np.float64)
                best_overflow = float("inf")
                best_mask = 0
                best_spread = float("inf")
                for mask in range(1, 1 << (m_act - 1)):
                    s0_sum = 0.0
                    s1_sum = 0.0
                    for bit in range(m_act):
                        if (mask >> bit) & 1:
                            s0_sum += loads[bit]
                        else:
                            s1_sum += loads[bit]
                    overflow = max(0.0, s0_sum - self.Q) + max(0.0, s1_sum - self.Q)
                    if overflow < best_overflow - 1e-6:
                        best_overflow = overflow
                        best_mask = mask
                        best_spread = abs(s0_sum - s1_sum)
                    elif abs(overflow - best_overflow) <= 1e-6:
                        spread = abs(s0_sum - s1_sum)
                        if spread < best_spread:
                            best_mask = mask
                            best_spread = spread
                bins: list[list[int]] = [[], []]
                for bit, store_i in enumerate(active_stores):
                    k_idx = 0 if ((best_mask >> bit) & 1) else 1
                    bins[k_idx].append(store_i)
                for k_idx in range(2):
                    b_sum = sum(q_t[i - 1] for i in bins[k_idx])
                    if b_sum > self.Q + 1e-7:
                        scale = self.Q / b_sum
                        for i in bins[k_idx]:
                            q_t[i - 1] *= scale
            else:
                sorted_stores = sorted(active_stores, key=lambda i: q_t[i - 1], reverse=True)
                bins = [[] for _ in range(self.K)]
                bin_loads = [0.0 for _ in range(self.K)]
                for i in sorted_stores:
                    load_i = q_t[i - 1]
                    best_k = int(np.argmin(bin_loads))
                    avail_cap = max(0.0, self.Q - bin_loads[best_k])
                    q_t[i - 1] = min(load_i, avail_cap)
                    if q_t[i - 1] > 1e-6:
                        bins[best_k].append(i)
                        bin_loads[best_k] += q_t[i - 1]

            # Ensure vehicle 0 is used before vehicle 1 (symmetry breaking constraint)
            non_empty_bins = [b for b in bins if len(b) > 0]
            bins = non_empty_bins + [[] for _ in range(self.K - len(non_empty_bins))]

            # Build 2-opt TSP tour for each active vehicle k
            for k in range(self.K):
                tour = bins[k]
                if not tour:
                    continue

                # Nearest-neighbor initialization from depot 0
                unvisited = list(tour)
                ordered = []
                curr = 0
                while unvisited:
                    nxt = min(unvisited, key=lambda j: self.dist[curr, j])
                    ordered.append(nxt)
                    unvisited.remove(nxt)
                    curr = nxt

                # 2-opt local improvement
                m = len(ordered)
                if m >= 3:
                    improved = True
                    while improved:
                        improved = False
                        nodes = [0] + ordered + [0]
                        for a_idx in range(1, m):
                            for b_idx in range(a_idx + 1, m + 1):
                                u1, u2 = nodes[a_idx - 1], nodes[a_idx]
                                v1, v2 = nodes[b_idx], nodes[b_idx + 1]
                                if self.dist[u1, v1] + self.dist[u2, v2] + 1e-9 < self.dist[u1, u2] + self.dist[v1, v2]:
                                    ordered[a_idx - 1 : b_idx] = reversed(ordered[a_idx - 1 : b_idx])
                                    improved = True
                                    break
                            if improved:
                                break

                # Set x, u, and q variables for vehicle k at period t
                z_int[self.x_indices[(0, ordered[0], k, t)]] = 1.0
                z_int[self.x_indices[(ordered[-1], 0, k, t)]] = 1.0
                for idx_pos, store_i in enumerate(ordered):
                    z_int[self.u_indices[(store_i, k, t)]] = float(idx_pos + 1)
                    z_int[self.q_indices[(store_i, k, t)]] = q_t[store_i - 1]
                    q_int_tot[store_i - 1, t] = q_t[store_i - 1]
                    if idx_pos + 1 < m:
                        z_int[self.x_indices[(store_i, ordered[idx_pos + 1], k, t)]] = 1.0

        # Simulate exact FIFO inventory flow under q_int_tot and demand (guaranteeing store capacity Ci)
        prev_I = np.zeros((self.N, self.tau - 1), dtype=np.float64)
        for t in range(self.T):
            for i in range(self.N):
                d_rem = float(demand[i, t])
                stock_age = np.zeros(self.tau, dtype=np.float64)
                stock_age[0] = q_int_tot[i, t]
                for a in range(1, self.tau):
                    stock_age[a] = prev_I[i, a - 1]

                w_i = np.zeros(self.tau, dtype=np.float64)
                for a in reversed(range(self.tau)):
                    if d_rem > 0.0:
                        used = min(stock_age[a], d_rem)
                        w_i[a] = used
                        d_rem -= used
                        stock_age[a] -= used

                # Check store storage capacity sum_{a=1}^{tau-1} I[i, t, a] <= Ci
                carryover = stock_age[: self.tau - 1]
                tot_carry = float(np.sum(carryover))
                if tot_carry > self.Ci + 1e-9:
                    # Trim fresh delivery q_int_tot[i, t] by excess carryover
                    excess = tot_carry - self.Ci
                    trim = min(stock_age[0], excess)
                    stock_age[0] -= trim
                    q_int_tot[i, t] -= trim
                    for k in range(self.K):
                        q_idx = self.q_indices[(i + 1, k, t)]
                        if z_int[q_idx] >= trim:
                            z_int[q_idx] -= trim
                            break

                for a in range(1, self.tau + 1):
                    z_int[self.w_indices[(i + 1, t, a)]] = w_i[a - 1]
                for a in range(1, self.tau):
                    z_int[self.I_indices[(i + 1, t, a)]] = stock_age[a - 1]
                    prev_I[i, a - 1] = stock_age[a - 1]

                z_int[self.y_pos_indices[(i + 1, t)]] = max(0.0, d_rem)
                z_int[self.y_neg_indices[(i + 1, t)]] = max(0.0, stock_age[self.tau - 1])

        obj_int = float(np.dot(c, z_int))
        return z_int, obj_int

    def solve(
        self,
        demand: np.ndarray,
        cost_vector: np.ndarray | None = None,
        time_limit_sec: float | None = None,
        mip_rel_gap: float = 0.05,
        use_milp: bool = False,
    ) -> PIRPSolution:
        """
        Solves the PIRP instance for the given demand matrix.

        Args:
            demand: Matrix of shape (N, T).
            cost_vector: Optional custom objective vector (for DFL/SPO+). Defaults to self.base_c.
            time_limit_sec: Solver time limit in seconds.
            mip_rel_gap: Requested relative MIP gap when ``use_milp`` is true.
            use_milp: Invoke HiGHS MILP when true; otherwise return a feasibility-checked
                constructive heuristic for discrete routing decisions.

        Returns:
            PIRPSolution containing extracted decisions, costs, and solver diagnostics.
        """
        import time
        start_t = time.perf_counter()

        c = np.array(cost_vector if cost_vector is not None else self.base_c, dtype=np.float64)
        A_sparse, b_l, b_u, num_eq = self.build_system(demand)

        if self.continuous_relaxation:
            # High-speed continuous LP solve via HiGHS using pre-built static matrices
            b_eq = b_u[:num_eq]

            options = {"disp": False}
            if time_limit_sec is not None:
                options["time_limit"] = time_limit_sec

            res = linprog(
                c=c,
                A_ub=self._A_ub,
                b_ub=self._b_ub,
                A_eq=self._A_eq,
                b_eq=b_eq,
                bounds=self._lp_bounds,
                method="highs",
                options=options,
            )
            elapsed = time.perf_counter() - start_t
            success = bool(res.success)
            z_sol = res.x if success else np.zeros(self.total_vars)
            obj_val = float(res.fun) if success else 0.0
            msg = res.message
            status = int(res.status)
            solver_name = "HiGHS LP"
            mip_gap = None
            mip_dual_bound = None
            mip_node_count = None
            is_optimal = success
            used_heuristic_fallback = False
        else:
            # The construction heuristic is a fast feasible candidate, not an optimality proof.
            z_heur, obj_heur = self._construct_integer_solution(demand, c)

            if use_milp:
                constraints = LinearConstraint(A_sparse, b_l, b_u)
                options = {"disp": False, "mip_rel_gap": mip_rel_gap}
                if time_limit_sec is not None:
                    options["time_limit"] = time_limit_sec
                res = milp(
                    c=c,
                    constraints=constraints,
                    bounds=self._milp_bounds,
                    integrality=self.integrality,
                    options=options,
                )
                has_sol = res.x is not None and np.all(np.isfinite(res.x))
                if has_sol:
                    z_milp = res.x
                    obj_milp = float(res.fun) if res.fun is not None else float(np.dot(c, z_milp))
                else:
                    z_milp = z_heur
                    obj_milp = float("inf")
                mip_gap = float(res.mip_gap) if getattr(res, "mip_gap", None) is not None else None
                mip_dual_bound = float(res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else None
                mip_node_count = int(res.mip_node_count) if getattr(res, "mip_node_count", None) is not None else None
                solver_name = "HiGHS MILP"
            else:
                z_milp = z_heur
                obj_milp = float("inf")
                res = None
                mip_gap = None
                mip_dual_bound = None
                mip_node_count = None
                solver_name = "LP-guided constructive heuristic"

            # Reject a purported incumbent if it violates a model constraint.
            def is_feasible(z: np.ndarray) -> bool:
                if not np.all(np.isfinite(z)):
                    return False
                lhs = A_sparse @ z
                if np.any(lhs < b_l - 1e-5) or np.any(lhs > b_u + 1e-5):
                    return False
                if np.any(z < self.lb - 1e-5) or np.any(z > self.ub + 1e-5):
                    return False
                if not self.continuous_relaxation and np.any(np.abs(z[self._x_idx_list] - np.rint(z[self._x_idx_list])) > 1e-5):
                    return False
                return True

            heuristic_feasible = is_feasible(z_heur)
            milp_feasible = use_milp and res is not None and res.x is not None and is_feasible(res.x)

            elapsed = time.perf_counter() - start_t
            prefer_heuristic = heuristic_feasible and (
                not milp_feasible
                or (res is not None and res.status != 0 and obj_heur <= obj_milp + 1e-8)
            )
            if prefer_heuristic:
                z_sol = z_heur
                obj_val = obj_heur
                success = True
                status = int(res.status) if res is not None else 1
                msg = "Feasible LP-guided constructive solution; optimality not certified"
                solver_name = (
                    "HiGHS MILP attempt + LP-guided feasible incumbent"
                    if res is not None
                    else "LP-guided constructive heuristic"
                )
                used_heuristic_fallback = True
                is_optimal = False
            else:
                z_sol = z_milp
                obj_val = obj_milp
                success = bool(milp_feasible)
                status = int(res.status) if res is not None else 4
                msg = str(res.message) if res is not None else "No feasible integer solution found"
                used_heuristic_fallback = False
                is_optimal = bool(milp_feasible and res.status == 0)

        # Vectorized extraction of structured decisions
        x_routes = np.zeros((self.N + 1, self.N + 1, self.K, self.T), dtype=np.float64)
        x_routes[self._x_i, self._x_j, self._x_k, self._x_t] = z_sol[self._x_idx_list]

        q_deliveries = z_sol[self._q_idx_arr]
        q_total = np.sum(q_deliveries, axis=1)  # (N, T)

        I_inventory = z_sol[self._I_idx_arr]
        w_consumed = z_sol[self._w_idx_arr]
        y_pos = z_sol[self._yp_idx_arr]
        y_neg = z_sol[self._yn_idx_arr]

        # Cost breakdown
        routing_cost = float(np.sum(x_routes * self.dist[:, :, None, None]))
        h_weights = np.array(
            [self.h + a_idx * self.aging_surcharge_eps for a_idx in range(self.tau - 1)],
            dtype=np.float64,
        )
        holding_cost = float(np.sum(I_inventory * h_weights))
        spoilage_cost = float(self.ps * np.sum(y_neg))
        stockout_cost = float(self.pu * np.sum(y_pos))
        total_cost = routing_cost + holding_cost + spoilage_cost + stockout_cost

        solution = PIRPSolution(
            success=success,
            status=status,
            message=msg,
            objective_val=obj_val,
            routing_cost=routing_cost,
            holding_cost=holding_cost,
            spoilage_cost=spoilage_cost,
            stockout_cost=stockout_cost,
            total_cost=total_cost,
            x_routes=x_routes,
            q_deliveries=q_deliveries,
            q_total=q_total,
            I_inventory=I_inventory,
            w_consumed=w_consumed,
            y_pos=y_pos,
            y_neg=y_neg,
            solve_time_sec=elapsed,
            is_lp_relaxation=self.continuous_relaxation,
            solver_name=solver_name,
            mip_gap=mip_gap,
            mip_dual_bound=mip_dual_bound,
            mip_node_count=mip_node_count,
            is_optimal=is_optimal,
            used_heuristic_fallback=used_heuristic_fallback,
        )
        self.solver_diagnostics.append({
            "success": bool(solution.success),
            "status": int(solution.status),
            "solver_name": solution.solver_name,
            "mip_gap": solution.mip_gap,
            "mip_dual_bound": solution.mip_dual_bound,
            "is_optimal": bool(solution.is_optimal),
            "used_heuristic_fallback": bool(solution.used_heuristic_fallback),
            "solve_time_sec": float(solution.solve_time_sec),
        })
        return solution


def solve_pirp(
    demand: np.ndarray,
    distance_matrix: np.ndarray | None = None,
    continuous_relaxation: bool = False,
    shelf_life_tau: int = 3,
    holding_cost_h: float = 1.0,
    aging_surcharge_eps: float = 0.01,
    stockout_penalty_pu: float = 20.0,
    spoilage_penalty_ps: float = 5.0,
    vehicle_capacity_Q: float = 100.0,
    store_capacity_Ci: float = 70.0,
    time_limit_sec: float | None = None,
    mip_rel_gap: float = 0.05,
    initial_inventory: np.ndarray | None = None,
    use_milp: bool = False,
) -> PIRPSolution:
    """Helper function to instantiate and solve a PIRP instance."""
    N, T = demand.shape
    model = PIRPModel(
        num_stores=N,
        horizon_T=T,
        distance_matrix=distance_matrix,
        continuous_relaxation=continuous_relaxation,
        initial_inventory=initial_inventory,
        shelf_life_tau=shelf_life_tau,
        holding_cost_h=holding_cost_h,
        aging_surcharge_eps=aging_surcharge_eps,
        stockout_penalty_pu=stockout_penalty_pu,
        spoilage_penalty_ps=spoilage_penalty_ps,
        vehicle_capacity_Q=vehicle_capacity_Q,
        store_capacity_Ci=store_capacity_Ci,
    )
    return model.solve(
        demand,
        time_limit_sec=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
        use_milp=use_milp,
    )
