"""Closed-loop perishable inventory-routing benchmark (revision 1).

Modules
-------
network      Solomon-derived geometry, exact subset TSP costs and exact CVRP packing.
demand       Contextual negative-binomial demand generator and feature construction.
simulator    Age-structured FIFO perishable inventory dynamics (NumPy and PyTorch).
milp         Rolling-horizon PIRP MILPs (deterministic and scenario-based) and recourse LP.
forecasting  LightGBM point and quantile forecasters.
policies     Inventory policies (BSP, BSP-EW, UL-lambda) and lookahead policies.
dfl          Decision-focused learning of planning targets.
evaluation   Closed-loop evaluation and paired statistics.
"""
