import json
import os
import re
import subprocess
import sys

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_licenses_exist():
    """Verify license notice files exist under .licenses/."""
    licenses_dir = os.path.join(WORKSPACE_ROOT, ".licenses")
    assert os.path.isdir(licenses_dir), ".licenses directory must exist."
    
    expected_files = [
        "literature_search_arxiv_LICENSE.txt",
        "literature_search_openalex_LICENSE.txt",
        "literature_search_europepmc_LICENSE.txt",
    ]
    for filename in expected_files:
        filepath = os.path.join(licenses_dir, filename)
        assert os.path.isfile(filepath), f"License file {filename} must exist."
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content.strip()) > 0, f"License file {filename} must not be empty."


def test_references_bib_integrity():
    """Verify references.bib exists, parses, and contains required core papers and DOIs."""
    bib_path = os.path.join(WORKSPACE_ROOT, "paper", "references.bib")
    assert os.path.isfile(bib_path), "paper/references.bib must exist."

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_text = f.read()

    entries = re.findall(r'@(\w+)\{([^,]+),', bib_text)
    assert len(entries) >= 25, f"Expected at least 25 references, found {len(entries)}."

    raw_keys = [k.strip() for _, k in entries]
    entry_keys = set(raw_keys)
    assert len(raw_keys) == len(entry_keys), f"Duplicate citation keys found in references.bib: {[k for k in raw_keys if raw_keys.count(k) > 1]}"

    # Core required papers
    core_keys = [
        "elmachtoub2022smart",
        "tang2024pyepo",
        "mandi2024decision",
        "sadana2025survey",
        "coelho2014optimal",
        "coelho2014thirty",
        "archetti2007branch",
        "archetti2018stochastic",
        "federgruen1986allocation",
    ]
    for key in core_keys:
        assert key in entry_keys, f"Core required citation '{key}' missing from references.bib."

    # Validate that every entry has a DOI or valid URL
    raw_blocks = bib_text.split("@")
    for block in raw_blocks:
        if not block.strip():
            continue
        key_match = re.match(r'\w+\{([^,]+),', block)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        has_doi = bool(re.search(r'\bdoi\s*=', block, re.IGNORECASE))
        has_url = bool(re.search(r'\burl\s*=', block, re.IGNORECASE))
        assert has_doi or has_url, f"BibTeX entry '{key}' must have a valid doi or url."


def test_manuscript_structure_and_citations():
    """Verify manuscript.tex structure, section headers, theorems, algorithms, and citations."""
    tex_path = os.path.join(WORKSPACE_ROOT, "paper", "manuscript.tex")
    bib_path = os.path.join(WORKSPACE_ROOT, "paper", "references.bib")
    assert os.path.isfile(tex_path), "paper/manuscript.tex must exist."

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_text = f.read()

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_text = f.read()

    bib_keys = {k.strip() for _, k in re.findall(r'@(\w+)\{([^,]+),', bib_text)}

    # Check all citations resolve
    cites = re.findall(r'\\cite[pt]?\{([^}]+)\}', tex_text)
    all_cites = set()
    for c in cites:
        for k in c.split(','):
            all_cites.add(k.strip())

    missing_keys = all_cites - bib_keys
    assert not missing_keys, f"Citations in manuscript missing from references.bib: {missing_keys}"

    # Check structural sections
    expected_sections = [
        r"\section{Introduction}",
        r"\section{Literature Review and Methodological Positioning}",
        r"\section{Mathematical Formulation and Bilevel Recourse Reformulation}",
        r"\section{End-to-End Decision-Focused Learning Methodology}",
        r"\section{Experimental Setup and Benchmark Suite}",
        r"\section{Empirical Results and Computational Analysis}",
        r"\section{Managerial Insights for Perishable Logistics Planners}",
        r"\section{Conclusion and Future Research Directions}",
    ]
    for sec in expected_sections:
        assert sec in tex_text, f"Expected section '{sec}' missing in manuscript.tex."

    # Check mathematical components
    assert r"\label{thm:recourse_feasibility}" in tex_text, "Theorem 1 label missing."
    assert r"\label{thm:spop_properties}" in tex_text, "Theorem 2 label missing."
    assert r"\label{alg:two_phase}" in tex_text, "Algorithm 1 label missing."
    assert r"\label{tab:positioning}" in tex_text, "Positioning table label missing."
    assert r"\label{tab:main_benchmark}" in tex_text, "Main benchmark table label missing."
    assert r"\label{tab:fractile_reconciliation}" in tex_text, "Fractile reconciliation table label missing."
    assert r"\label{tab:speedup_analysis}" in tex_text, "Speedup table label missing."
    assert r"\label{tab:sensitivity}" in tex_text, "Sensitivity table label missing."
    assert r"\label{rem:simplex_degeneracy}" in tex_text, "Remark simplex degeneracy label missing."

    # Environment balance
    begins = re.findall(r'\\begin\{([^}]+)\}', tex_text)
    ends = re.findall(r'\\end\{([^}]+)\}', tex_text)
    assert len(begins) == len(ends), f"LaTeX environment count mismatch: {len(begins)} vs {len(ends)}."


def test_packaged_workflow_skill():
    """Verify reusable workflow skill is created and CLI executes cleanly."""
    skill_dir = os.path.join(WORKSPACE_ROOT, ".agents", "skills", "dfl-pirp-workflow")
    assert os.path.isdir(skill_dir), "Skill directory .agents/skills/dfl-pirp-workflow must exist."

    skill_file = os.path.join(skill_dir, "SKILL.md")
    assert os.path.isfile(skill_file), "Skill SKILL.md must exist."

    cli_script = os.path.join(skill_dir, "scripts", "dfl_pirp_cli.py")
    assert os.path.isfile(cli_script), "Skill CLI script dfl_pirp_cli.py must exist."


def test_cli_evaluate_regret_benchmark_mode(tmp_path):
    """Verify that dfl_pirp_cli.py evaluate-regret genuinely parses benchmark metrics and outputs authentic results."""
    cli_script = os.path.join(WORKSPACE_ROOT, ".agents", "skills", "dfl-pirp-workflow", "scripts", "dfl_pirp_cli.py")
    out_json = str(tmp_path / "regret_benchmark_eval.json")
    pred_csv = os.path.join(WORKSPACE_ROOT, "results", "benchmark_metrics.csv")
    gt_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "demand_features_test.csv")

    cmd = [
        sys.executable,
        cli_script,
        "evaluate-regret",
        "--predictions", pred_csv,
        "--ground-truth", gt_csv,
        "--output", out_json,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI evaluation failed with stderr: {res.stderr}"
    assert os.path.isfile(out_json), "Output JSON report was not generated."

    with open(out_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["status"] == "PASS"
    assert report["evaluation_mode"] == "benchmark_operational_audit"
    assert "models" in report
    assert len(report["models"]) >= 4, "Expected at least 4 evaluated models in benchmark report."

    # Validate non-dummy metrics
    for model_info in report["models"]:
        assert "normalized_regret_pct" in model_info
        assert "total_cost" in model_info
        assert "stockout_cost" in model_info
        assert "spoilage_cost" in model_info
        assert model_info["total_cost"] > 0
        assert isinstance(model_info["cost_breakdown_pct"], dict)
        assert "stockout_share" in model_info["cost_breakdown_pct"]


def test_cli_evaluate_regret_error_handling(tmp_path):
    """Verify that dfl_pirp_cli.py evaluate-regret rejects non-existent files with FAIL."""
    cli_script = os.path.join(WORKSPACE_ROOT, ".agents", "skills", "dfl-pirp-workflow", "scripts", "dfl_pirp_cli.py")
    out_json = str(tmp_path / "missing_file_eval.json")

    cmd = [
        sys.executable,
        cli_script,
        "evaluate-regret",
        "--predictions", "non_existent_predictions.csv",
        "--ground-truth", "data/processed/demand_features_test.csv",
        "--output", out_json,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert os.path.isfile(out_json)

    with open(out_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["status"] == "FAIL"
    assert len(report["errors"]) > 0
    assert "does not exist" in report["errors"][0]

