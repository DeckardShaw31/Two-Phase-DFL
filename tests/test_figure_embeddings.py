import os
import re

import pandas as pd

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "results", "figures")
MANUSCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "paper", "manuscript.tex")
MULTI_SEED_PATH = os.path.join(WORKSPACE_ROOT, "results", "multi_seed_summary.csv")
CAPACITY_DIAG_PATH = os.path.join(WORKSPACE_ROOT, "results", "capacity_diagnostics.csv")

EXPECTED_FIGURES = [
    "fig_regret_comparison.png",
    "fig_convergence_curves.png",
    "fig_speedup_pareto.png",
    "fig_sensitivity_heatmap.png",
    "fig_inventory_trajectory.png",
]

EXPECTED_LABELS = [
    "fig:regret_comparison",
    "fig:convergence_curves",
    "fig:speedup_pareto",
    "fig:sensitivity_heatmap",
    "fig:inventory_trajectory",
]


def test_all_five_figure_pngs_exist():
    """Verify that all 5 publication-grade figure PNG files exist on disk with valid PNG headers."""
    assert os.path.isdir(RESULTS_FIGURES_DIR), f"Figures directory {RESULTS_FIGURES_DIR} must exist."
    for fig_name in EXPECTED_FIGURES:
        fig_path = os.path.join(RESULTS_FIGURES_DIR, fig_name)
        assert os.path.isfile(fig_path), f"Figure file '{fig_name}' must exist in {RESULTS_FIGURES_DIR}."
        file_size = os.path.getsize(fig_path)
        assert file_size > 10000, f"Figure '{fig_name}' file size ({file_size} bytes) is suspiciously small."
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"Figure '{fig_name}' must be a valid PNG binary."


def test_manuscript_includes_all_five_figures():
    """Verify that paper/manuscript.tex includes all 5 figure files via \\includegraphics and has their labels."""
    assert os.path.isfile(MANUSCRIPT_PATH), f"Manuscript file {MANUSCRIPT_PATH} must exist."
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        tex = f.read()

    for fig_name in EXPECTED_FIGURES:
        stem = os.path.splitext(fig_name)[0]
        # Match either the full filename or stem in includegraphics
        pattern = rf"\\includegraphics\[[^\]]*\]\{{[^}}]*{stem}[^}}]*\}}"
        match = re.search(pattern, tex)
        assert match is not None, f"Manuscript must contain \\includegraphics embedding for '{stem}'."

    for label in EXPECTED_LABELS:
        assert rf"\label{{{label}}}" in tex, f"Manuscript must define \\label{{{label}}}."


def test_manuscript_cites_all_five_figures_in_prose():
    """Verify that each figure label is referenced in the prose via \\ref{fig:...}."""
    assert os.path.isfile(MANUSCRIPT_PATH), f"Manuscript file {MANUSCRIPT_PATH} must exist."
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        tex = f.read()

    for label in EXPECTED_LABELS:
        # Check that \\ref{label} appears outside of \\label{label}
        refs = re.findall(rf"\\ref\{{{label}\}}", tex)
        assert len(refs) >= 1, f"Figure label '{label}' must be cited in the prose via \\ref{{{label}}}."


def test_physical_parameters_specified_in_section_5():
    """Verify Section 5.1 of manuscript explicitly specifies all physical network parameters."""
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        tex = f.read()

    # Required parameter specifications
    required_specs = [
        "N = 10",
        "T = 5",
        "K = 2",
        "Q = 100",
        "200",  # K * Q = 200
        "C_i = 70",
        "C_0 = 1{,}000",  # or 1,000 or 1000
        "I_{i,0,a} = 0",
        "h = 1.0",
        "p_s = 5",
        "p_u",
        "\\gamma \\in \\{0.8, 1.0, 1.5\\}",
    ]
    for spec in required_specs:
        assert spec in tex, f"Required physical parameter specification '{spec}' missing in manuscript."


def test_numerical_parity_with_multi_seed_summary():
    """Verify 100% numerical parity between Table 1/prose in manuscript and multi_seed_summary.csv."""
    assert os.path.isfile(MULTI_SEED_PATH), "results/multi_seed_summary.csv must exist."
    df = pd.read_csv(MULTI_SEED_PATH)
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        tex = f.read()

    ours = df[df["Model"] == "Two-Phase_DFL (Ours)"].iloc[0]
    regret_val = f"{ours['Regret_Mean_pct']:.2f}"
    cost_val = f"{ours['TotalCost_Mean']:.2f}"
    stockout_val = f"{ours['StockoutCost_Mean']:.2f}"

    assert regret_val in ["22.21", "22.28"], f"Expected Two-Phase DFL regret ~22.2%, got {regret_val}"
    assert cost_val in ["2126.03", "2143.60"], f"Expected Two-Phase DFL total cost ~2126-2144, got {cost_val}"
    assert stockout_val in ["1468.88", "1483.26"], f"Expected Two-Phase DFL stockout cost ~1468-1484, got {stockout_val}"

    # Check these appear in manuscript
    assert "22.21" in tex or "22.28" in tex, "Manuscript must report Two-Phase DFL regret."
    assert any(c in tex for c in ["2126.03", "2{,}126.03", "2143.60", "2{,}143.60"]), "Manuscript must report total cost."
    assert any(s in tex for s in ["1468.88", "1{,}468.88", "1483.26", "1{,}483.26"]), "Manuscript must report stockout cost."


def test_capacity_diagnostics_numerical_parity():
    """Verify that capacity diagnostics numbers match capacity_diagnostics.csv."""
    assert os.path.isfile(CAPACITY_DIAG_PATH), "results/capacity_diagnostics.csv must exist."
    df = pd.read_csv(CAPACITY_DIAG_PATH)
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        tex = f.read()

    ours = df[df["Model"] == "Two-Phase_DFL (Ours)"]
    mean_binding = ours["Capacity_Binding_Freq"].mean() * 100
    mean_stockout = ours["Realized_Stockout"].mean()

    assert any(abs(mean_binding - target) < 0.1 for target in [68.666667, 69.333333]), f"Expected mean binding freq ~68.7% or 69.3%, got {mean_binding}"
    assert any(abs(mean_stockout - target) < 0.05 for target in [14.696474, 14.840280]), f"Expected mean realized stockout ~14.70 or 14.84 units, got {mean_stockout}"

    assert any(p in tex for p in ["68.7\\%", "68.7%", "69.3\\%", "69.3%"]), "Manuscript must report mean binding frequency."
    assert any(p in tex for p in ["46.7\\%", "46.7%", "50.0\\%", "50.0%"]), "Manuscript must report Period 3 binding frequency."
    assert any(s in tex for s in ["14.70", "14.84"]), "Manuscript must report realized stockout."
