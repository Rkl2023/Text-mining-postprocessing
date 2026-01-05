"""
Lightweight validator for pinned dependencies.

Reads requirements.txt, checks installed versions, and reports any
missing or mismatched packages. Intended to keep PolyStruct-Mine
environments deterministic for reproducible runs.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping from requirement names to importable module names when they differ.
MODULE_ALIASES: Dict[str, str] = {
    "rdkit-pypi": "rdkit",
    "pyyaml": "yaml",
    "sqlalchemy": "sqlalchemy",
    "scikit-learn": "sklearn",
}


def parse_requirements(req_path: Path) -> List[Tuple[str, str]]:
    requirements: List[Tuple[str, str]] = []
    for raw_line in req_path.read_text().splitlines():
        # Strip inline comments and whitespace.
        line = re.sub(r"#.*", "", raw_line).strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        requirements.append((name.strip(), version.strip()))
    return requirements


def check_requirement(name: str, expected_version: str) -> Tuple[bool, str]:
    dist_name = name
    module_name = MODULE_ALIASES.get(name.lower(), name.replace("-", "_"))
    try:
        installed_version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return False, "missing"

    if installed_version != expected_version:
        return False, f"version mismatch (installed {installed_version}, expected {expected_version})"

    # Attempt import to verify module availability.
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"installed but import failed: {exc}"

    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check installed packages against requirements.txt")
    parser.add_argument(
        "-r",
        "--requirements",
        default=str(Path(__file__).resolve().parents[1] / "requirements.txt"),
        help="Path to requirements.txt",
    )
    args = parser.parse_args()

    req_path = Path(args.requirements)
    if not req_path.exists():
        print(f"[ERROR] requirements file not found: {req_path}", file=sys.stderr)
        return 1

    requirements = parse_requirements(req_path)
    if not requirements:
        print("[WARN] No requirements parsed; ensure the file is not empty.")
        return 0

    failures = []
    for name, version in requirements:
        ok, detail = check_requirement(name, version)
        status = "OK" if ok else "FAIL"
        print(f"{status:4s} {name} == {version} ({detail})")
        if not ok:
            failures.append((name, detail))

    if failures:
        print(f"\n[SUMMARY] {len(failures)} issue(s) detected.")
        for name, detail in failures:
            print(f" - {name}: {detail}")
        return 1

    print("\n[SUMMARY] All requirements satisfied.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
