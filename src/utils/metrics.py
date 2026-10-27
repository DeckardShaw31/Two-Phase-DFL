"""Evaluation metrics for Perishable Inventory-Routing: Decision Regret and Cost Breakdown."""
import numpy as np


def simulate_perishable_inventory_fifo(
    q_delivered: np.ndarray,
    true_demand: np.ndarray,
    tau: int = 3,
    initial_inventory: np.ndarray | None = None,
    store_capacity_Ci: float = float("inf"),
    return_components: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Simulates exact FIFO perishable inventory dynamics at retail stores under true demand.
    Enforces store physical storage capacity Ci via oldest-first overflow disposal (Algorithm 1).

    Args:
        q_delivered: Replenishment quantity delivered to each store (N, T).
        true_demand: True customer demand realization (N, T).
        tau: Shelf life (in periods).
        initial_inventory: Initial stock of age 1..tau-1 (N, tau-1), defaults to zeros.
        store_capacity_Ci: Maximum physical store storage capacity.

    Returns:
        I_on_hand: Carryover inventory at end of each period (N, T, tau-1).
        y_pos: Stockout / unmet emergency demand (N, T).
        y_neg: Spoilage / expired / discarded inventory (N, T).
        w_consumed: Inventory consumed by age (N, T, tau).
    """
    N, T = true_demand.shape
    I_on_hand = np.zeros((N, T, tau - 1), dtype=np.float64)
    y_pos = np.zeros((N, T), dtype=np.float64)
    y_expired = np.zeros((N, T), dtype=np.float64)
    y_capacity_discard = np.zeros((N, T), dtype=np.float64)
    w_consumed = np.zeros((N, T, tau), dtype=np.float64)

    # Current inventory at start of period: for ages 1..tau
    # At t=0, incoming stock before delivery is initial_inventory (ages 1..tau-1)
    prev_I = np.zeros((N, tau - 1), dtype=np.float64)
    if initial_inventory is not None:
        prev_I = np.array(initial_inventory, dtype=np.float64)
        if prev_I.shape != (N, tau - 1):
            raise ValueError(f"initial_inventory must have shape ({N}, {tau - 1})")
        if np.any(prev_I < 0) or np.any(~np.isfinite(prev_I)):
            raise ValueError("initial_inventory must contain finite nonnegative quantities")

    for t in range(T):
        for i in range(N):
            demand_remaining = float(true_demand[i, t])

            # Available stock entering period t:
            # - Newly delivered stock arrives at age 1
            # - Previous inventory aged by 1 period: age a-1 becomes age a
            stock_by_age = np.zeros(tau, dtype=np.float64)
            stock_by_age[0] = max(0.0, float(q_delivered[i, t]))
            for a in range(1, tau):
                stock_by_age[a] = max(0.0, float(prev_I[i, a - 1]))

            # FIFO Consumption: oldest inventory consumed first (from tau down to 1)
            for a in reversed(range(tau)):
                avail = stock_by_age[a]
                if demand_remaining > 0:
                    consumed = min(avail, demand_remaining)
                    w_consumed[i, t, a] = consumed
                    demand_remaining -= consumed
                    stock_by_age[a] -= consumed

            # Recourse:
            # Unmet demand
            y_pos[i, t] = max(0.0, demand_remaining)

            # Oldest unconsumed inventory (age tau) expires / spoils
            y_expired[i, t] = max(0.0, stock_by_age[tau - 1])

            # Store capacity enforcement (Algorithm 1):
            # If tentative surviving intermediate inventory exceeds store_capacity_Ci,
            # discard excess inventory from oldest intermediate cohort first (a = tau-1 down to 1)
            tentative_carryover = float(np.sum(stock_by_age[: tau - 1]))
            excess = max(0.0, tentative_carryover - store_capacity_Ci)
            if excess > 0.0:
                for a in reversed(range(tau - 1)):
                    if excess <= 0.0:
                        break
                    discard = min(stock_by_age[a], excess)
                    stock_by_age[a] -= discard
                    excess -= discard
                    y_capacity_discard[i, t] += discard

            # Remaining unconsumed inventory for ages 1..tau-1 carries over
            for a in range(tau - 1):
                I_on_hand[i, t, a] = stock_by_age[a]

            prev_I[i, :] = I_on_hand[i, t, :]

    if return_components:
        return I_on_hand, y_pos, y_expired, y_capacity_discard, w_consumed
    return I_on_hand, y_pos, y_expired + y_capacity_discard, w_consumed


def evaluate_decisions_under_true_demand(
    q_plan: np.ndarray,
    x_plan: np.ndarray,
    true_demand: np.ndarray,
    distance_matrix: np.ndarray,
    h: float = 1.0,
    aging_surcharge_eps: float = 0.01,
    p_s: float = 5.0,
    p_u: float = 20.0,
    tau: int = 3,
    initial_inventory: np.ndarray | None = None,
    store_capacity_Ci: float = 70.0,
    vehicle_capacity_Q: float = 100.0,
) -> dict[str, float]:
    """
    Evaluates the exact realized operational costs of planned decisions under true demand.

    Args:
        q_plan: Planned deliveries (N, T) or (K, N, T).
        x_plan: Planned vehicle routes (N+1, N+1, K, T) or total routing cost.
        true_demand: True customer demand (N, T).
        distance_matrix: (N+1, N+1) distance matrix.
        h: Base holding cost per unit per period for fresh carryover stock (a=1).
        aging_surcharge_eps: Marginal surcharge per unit age increment, h_a = h + (a-1)*eps.
        p_s: Spoilage penalty per unit.
        p_u: Stockout penalty per unit.
        tau: Shelf life.
        initial_inventory: (N, tau-1) initial stock.
        store_capacity_Ci: Store storage capacity limit.

    Returns:
        Dictionary with routing_cost, holding_cost, spoilage_cost, stockout_cost, total_cost.
    """
    # If q_plan has shape (K, N, T), sum over vehicles K
    if q_plan.ndim == 3 and q_plan.shape[0] != true_demand.shape[0]:
        q_delivered = np.sum(q_plan, axis=0)
    else:
        q_delivered = np.array(q_plan, dtype=np.float64)

    # Compute routing cost from x_plan
    if isinstance(x_plan, (int, float, np.floating)):
        routing_cost = float(x_plan)
    elif x_plan.ndim == 4:
        # x_plan shape: (N+1, N+1, K, T)
        routing_cost = float(np.sum(x_plan * distance_matrix[:, :, None, None]))
    elif x_plan.ndim == 3:
        # (N+1, N+1, T)
        routing_cost = float(np.sum(x_plan * distance_matrix[:, :, None]))
    else:
        routing_cost = 0.0

    I_on_hand, y_pos, y_expired, y_capacity_discard, _ = simulate_perishable_inventory_fifo(
        q_delivered=q_delivered,
        true_demand=true_demand,
        tau=tau,
        initial_inventory=initial_inventory,
        store_capacity_Ci=store_capacity_Ci,
        return_components=True,
    )

    # Age-differentiated holding cost: h_a = h + (a - 1) * aging_surcharge_eps
    h_weights = np.array(
        [h + a_idx * aging_surcharge_eps for a_idx in range(tau - 1)],
        dtype=np.float64,
    )
    holding_cost = float(np.sum(I_on_hand * h_weights))
    expired_units = float(np.sum(y_expired))
    capacity_discard_units = float(np.sum(y_capacity_discard))
    spoilage_cost = float(p_s * (expired_units + capacity_discard_units))
    stockout_cost = float(p_u * np.sum(y_pos))
    total_cost = routing_cost + holding_cost + spoilage_cost + stockout_cost

    total_demand = float(np.sum(true_demand))
    unmet_units = float(np.sum(y_pos))
    delivery_units = float(np.sum(q_delivered))
    if isinstance(x_plan, np.ndarray) and x_plan.ndim == 4:
        route_distance = routing_cost
        routes_used = float(np.sum(x_plan[0, 1:, :, :]))
    elif isinstance(x_plan, np.ndarray) and x_plan.ndim == 3:
        route_distance = routing_cost
        routes_used = float(np.sum(x_plan[0, 1:, :]))
    else:
        route_distance = routing_cost
        routes_used = 0.0
    fleet_capacity_utilization = delivery_units / (routes_used * vehicle_capacity_Q) if routes_used > 0 else 0.0

    result: dict[str, float] = {
        "routing_cost": routing_cost,
        "route_distance": route_distance,
        "routes_used": routes_used,
        "fleet_capacity_utilization": float(fleet_capacity_utilization),
        "holding_cost": holding_cost,
        "spoilage_cost": spoilage_cost,
        "expiry_cost": float(p_s * expired_units),
        "capacity_disposal_cost": float(p_s * capacity_discard_units),
        "expired_units": expired_units,
        "capacity_discard_units": capacity_discard_units,
        "stockout_cost": stockout_cost,
        "total_cost": total_cost,
        "total_demand_units": total_demand,
        "unmet_demand_units": unmet_units,
        "fill_rate": 1.0 - unmet_units / total_demand if total_demand > 0 else 1.0,
        "delivery_units": delivery_units,
        "on_hand_unit_periods": float(np.sum(I_on_hand)),
        "average_on_hand_per_store_period": float(np.mean(np.sum(I_on_hand, axis=2))),
        "ending_inventory_units": float(np.sum(I_on_hand[:, -1, :])),
    }
    for age_idx in range(tau - 1):
        result[f"average_on_hand_age_{age_idx + 1}"] = float(np.mean(I_on_hand[:, :, age_idx]))
    for period_idx in range(true_demand.shape[1]):
        period_demand = float(np.sum(true_demand[:, period_idx]))
        result[f"fill_rate_period_{period_idx + 1}"] = (
            1.0 - float(np.sum(y_pos[:, period_idx])) / period_demand if period_demand > 0 else 1.0
        )
        for age_idx in range(tau - 1):
            result[f"average_on_hand_age_{age_idx + 1}_period_{period_idx + 1}"] = float(
                np.mean(I_on_hand[:, period_idx, age_idx])
            )
    for store_idx in range(true_demand.shape[0]):
        store_demand = float(np.sum(true_demand[store_idx, :]))
        result[f"fill_rate_store_{store_idx + 1}"] = (
            1.0 - float(np.sum(y_pos[store_idx, :])) / store_demand if store_demand > 0 else 1.0
        )
    return result


# Standard alias matching paper and research documentation terminology
evaluate_fifo_pipeline = evaluate_decisions_under_true_demand


def compute_decision_regret(
    model_cost: float,
    oracle_cost: float,
) -> tuple[float, float]:
    """
    Computes absolute regret and normalized percentage regret.

    Args:
        model_cost: Total operational cost incurred by the model's decision under true demand.
        oracle_cost: Clairvoyant optimal cost under perfect foresight.

    Returns:
        (abs_regret, normalized_regret_pct)
    """
    abs_regret = max(0.0, model_cost - oracle_cost)
    if oracle_cost > 1e-6:
        norm_regret = (abs_regret / oracle_cost) * 100.0
    else:
        norm_regret = 0.0
    return abs_regret, norm_regret


def compute_forecast_accuracy(
    pred_demand: np.ndarray,
    true_demand: np.ndarray,
) -> dict[str, float]:
    """
    Computes statistical forecast accuracy metrics (MSE, MAE, RMSE).
    """
    mse = float(np.mean((pred_demand - true_demand) ** 2))
    mae = float(np.mean(np.abs(pred_demand - true_demand)))
    rmse = float(np.sqrt(mse))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
