# Historical experiment scripts

The following scripts belong to the original submission workflow and are retained for auditability. They do not implement the corrected independent-episode protocol, and their generated tables/figures are not evidence for the revised manuscript:

- `run_benchmark.py`
- `run_multi_seed_benchmark.py`
- `run_sensitivity.py`
- `run_statistical_tests.py`
- `run_lp_warmup.py`
- `run_capacity_diagnostics.py`
- `reproduce_all.py`
- `audit_manuscript.py`

Use `run_revision_benchmark.py` for the corrected, narrow synthetic benchmark. Even that run does not establish state-of-the-art coverage: its Crama UL-lambda policy is an open-loop adaptation, and Jia et al.'s scenario PTO and He et al.'s robust perishable PIRP are not implemented.

Reasons the legacy outputs were withdrawn include dependent adjacent demand blocks, within-horizon feature leakage relative to episode-start plans, M5 provenance claims unsupported by the generator, solver outputs described as optimal without adequate evidence, regret against heuristic plans, and comparator labels that did not match their implementations.
