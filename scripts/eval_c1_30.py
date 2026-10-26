import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.baselines import LightGBMPTO, RidgePTO
from src.optimization.pirp_model import solve_pirp
from src.utils.metrics import compute_decision_regret, evaluate_decisions_under_true_demand

df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset()
N, T = config['num_stores'], config['horizon_T']
ps = config['spoilage_penalty_ps']
h = config['holding_cost_h']
Ci = config['store_capacity_Ci']
Q = 100.0
tau = 3

X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)
X_train_flat = df_train[feat_cols].values
y_train_flat = df_train['demand'].values

num_test = len(X_test_3d)

ridge = RidgePTO(alpha=10.0).fit(X_train_flat, y_train_flat)
lgb = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_train_flat, y_train_flat)

preds_ridge = np.array([ridge.predict(X_test_3d[i].reshape(N*T, -1)).reshape(N, T) for i in range(num_test)])
preds_lgb = np.array([lgb.predict(X_test_3d[i].reshape(N*T, -1)).reshape(N, T) for i in range(num_test)])

for pu in [10.0, 20.0, 50.0]:
    print(f"=== Evaluating pu/h = {pu} on {num_test} instances ===")
    oracle_costs = []
    ridge_metrics = []
    lgb_metrics = []
    
    for i in range(num_test):
        d_true = d_test_3d[i]
        sol_oracle = solve_pirp(d_true, dist_matrix, continuous_relaxation=False, shelf_life_tau=tau, stockout_penalty_pu=pu, spoilage_penalty_ps=ps, vehicle_capacity_Q=Q, store_capacity_Ci=Ci, time_limit_sec=2.0)
        c_oracle = sol_oracle.total_cost
        oracle_costs.append(c_oracle)
        
        sol_r = solve_pirp(preds_ridge[i], dist_matrix, continuous_relaxation=False, shelf_life_tau=tau, stockout_penalty_pu=pu, spoilage_penalty_ps=ps, vehicle_capacity_Q=Q, store_capacity_Ci=Ci, time_limit_sec=2.0)
        m_r = evaluate_decisions_under_true_demand(sol_r.q_total, sol_r.x_routes, d_true, dist_matrix, h=h, p_s=ps, p_u=pu, tau=tau)
        _, reg_r = compute_decision_regret(m_r['total_cost'], c_oracle)
        m_r['regret'] = reg_r
        ridge_metrics.append(m_r)
        
        sol_l = solve_pirp(preds_lgb[i], dist_matrix, continuous_relaxation=False, shelf_life_tau=tau, stockout_penalty_pu=pu, spoilage_penalty_ps=ps, vehicle_capacity_Q=Q, store_capacity_Ci=Ci, time_limit_sec=2.0)
        m_l = evaluate_decisions_under_true_demand(sol_l.q_total, sol_l.x_routes, d_true, dist_matrix, h=h, p_s=ps, p_u=pu, tau=tau)
        _, reg_l = compute_decision_regret(m_l['total_cost'], c_oracle)
        m_l['regret'] = reg_l
        lgb_metrics.append(m_l)
        
    print(f"Oracle mean cost: {np.mean(oracle_costs):.2f}")
    r_reg = np.mean([m['regret'] for m in ridge_metrics])
    r_tot = np.mean([m['total_cost'] for m in ridge_metrics])
    r_spoil = np.mean([m['spoilage_cost'] for m in ridge_metrics])
    r_stock = np.mean([m['stockout_cost'] for m in ridge_metrics])
    print(f"Ridge: Regret={r_reg:.2f}%, TotalCost={r_tot:.2f}, Spoilage={r_spoil:.2f}, Stockout={r_stock:.2f}")

    l_reg = np.mean([m['regret'] for m in lgb_metrics])
    l_tot = np.mean([m['total_cost'] for m in lgb_metrics])
    l_spoil = np.mean([m['spoilage_cost'] for m in lgb_metrics])
    l_stock = np.mean([m['stockout_cost'] for m in lgb_metrics])
    print(f"LightGBM: Regret={l_reg:.2f}%, TotalCost={l_tot:.2f}, Spoilage={l_spoil:.2f}, Stockout={l_stock:.2f}")
