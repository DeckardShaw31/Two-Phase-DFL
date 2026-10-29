"""Build the anonymised replication package (supplementary material) from the working tree.

The archive holds the code, tests, result files, the dataset card, README, licence and pinned
requirements, plus the generated tables and figures of the manuscript under manuscript_tables/.
It never contains the git history, the manuscript sources, letters or notes, the data cache or
the FreshRetailNet data files. Files are taken from the working tree (tracked or new, not
ignored), so uncommitted changes are included. The build stops if any file contains a string
that identifies the author.

python -m experiments.package_replication --out supplementary_material/replication_package.zip
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import zipfile

ROOT_FILES = {"README.md", "LICENSE", "requirements.txt", "pyproject.toml", "data/freshretailnet_raw/README.md"}
DIRS = ("pirp/", "experiments/", "tests/", "results_v2/", "results/")
EXCLUDE = {"experiments/package_replication.py"}  # packaging tool, not part of the replication
TABLES = "revision_1/figures/"  # generated tables and figures -> manuscript_tables/
GENERATED = re.compile(r"^(numbers_auto\.tex|tab_.*\.tex|fig_.*\.pdf|sensitivity_long\.csv)$")
IDENTIFYING = re.compile(rb"proin|thanh|nguyen|nguy\xe1\xbb\x85n|hcmiu|deckardshaw|orcid", re.I)


def working_tree_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(f for f in out.split("\0") if f and os.path.isfile(f))


def selection() -> dict[str, str]:
    """Archive name -> source path."""
    sel = {}
    for f in working_tree_files():
        if f in EXCLUDE or "__pycache__" in f:
            continue
        if f in ROOT_FILES or f.startswith(DIRS):
            sel[f] = f
        elif f.startswith(TABLES) and GENERATED.match(f[len(TABLES):]):
            sel["manuscript_tables/" + f[len(TABLES):]] = f
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplementary_material/replication_package.zip")
    ap.add_argument("--top", default="replication_package", help="top-level folder inside the archive")
    a = ap.parse_args()
    sel = selection()
    hits = [src for src in sel.values() if IDENTIFYING.search(open(src, "rb").read())]
    if hits:
        raise SystemExit("identifying strings found, nothing written:\n  " + "\n  ".join(hits))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, src in sorted(sel.items()):
            z.write(src, f"{a.top}/{arc}")
    size = sum(os.path.getsize(s) for s in sel.values())
    print(f"{len(sel)} files, {size / 1e6:.1f} MB uncompressed, {os.path.getsize(a.out) / 1e6:.1f} MB archive -> {a.out}")


if __name__ == "__main__":
    main()
