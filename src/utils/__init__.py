"""Utility modules: evaluation metrics and publication plotting."""
from .metrics import (
    compute_decision_regret,
    compute_forecast_accuracy,
    evaluate_decisions_under_true_demand,
    simulate_perishable_inventory_fifo,
)
from .plotting import (
    generate_all_publication_plots,
    plot_inventory_trajectory,
    plot_regret_comparison,
    plot_sensitivity_heatmap,
    plot_speedup_pareto,
)

__all__ = [
    "compute_decision_regret",
    "compute_forecast_accuracy",
    "evaluate_decisions_under_true_demand",
    "generate_all_publication_plots",
    "plot_inventory_trajectory",
    "plot_regret_comparison",
    "plot_sensitivity_heatmap",
    "plot_speedup_pareto",
    "simulate_perishable_inventory_fifo",
]
