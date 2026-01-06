PolyStruct-Mine: Polymer Structure Mining Pipeline
=================================================

Checklist (quick start)
- Prereqs: Python 3.10, Java (OpenJDK 17+ recommended), RDKit, JPype1
- Install: `conda create -n polystruct-mine python=3.10 -y && conda activate polystruct-mine && pip install -r requirements.txt`
- OPSIN: build or use bundled `OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar`; ensure Java available; JPype1 loads the JAR
- Run UI: `streamlit run ui_app.py` (uses cached inputs; does not auto-run pipeline)
- Run pipeline: `python -m polymer_pipeline.pipeline --text_table data/example_polymer_data.csv --image_pool data/example_image_candidates.csv`
- Verify: run OPSIN smoke test (below) and RDKit draw test

Project overview
----------------
PolyStruct-Mine extracts polymer structural information from text-mined names (OPSIN + heuristics) and image-derived candidates. It standardizes backbone and side-chain representations, matches text rows to image candidates, and outputs repeat-unit SMILES with provenance, confidence, and audit logs.

Pipeline architecture (Layers 0–4)
1. Layer 0 — Standardization: Normalize schema/columns, clean names/SMILES, log schema report.
2. Layer 1 — Name → Structure: OPSIN (via JPype + local JAR) plus domain dictionaries/alkyl heuristics to derive backbone/sidechain SMILES from text.
3. Layer 2 — Candidate Matching: Join text rows to image-pool SMILES by paper_id; RDKit-validate, score, and select best candidate; derive repeat-unit and Murcko scaffold.
4. Layer 3 — Conflict Resolution: Finalize SMILES fields, assign confidence/decision notes.
5. Layer 4 — Manual Review Minimization: Flag entries needing review; produce metrics and logs.

Inputs
------
Text-mined polymer table (polymer-centric)
- Required: `paper_id`, `polymer_name`, `backbone_name_text`, `sidechain_name_text`
- Optional: `sidechain_type`, `sidechain_length`, `device_type`, `confidence_text`
- Sample: `examples/text_polymers_sample.csv`

Image-recognition candidate pool (fragment-centric)
- Required: `paper_id`, `image_id`, `fragment_id`, `candidate_smiles`, `parse_status`
- Optional features: `has_metal`, `has_counterion`, `eg_motif_count`, `max_aliphatic_chain_len`, `heteroatom_counts`, `heavy_atom_count`
- Sample: `examples/image_pool_sample.csv`

Outputs
-------
Generated file: `polymer_structures_groundtruth.csv` with columns:
```
paper_id, polymer_name, polymer_name_norm,
repeat_unit_smiles_full, backbone_smiles, sidechain_smiles_list,
source_priority, confidence, needs_manual_review, decision_note
```
Provenance fields (e.g., matching_decision, candidate_score) are retained for auditability.

Installation
------------
Recommended: conda on macOS/Linux; Python 3.10 (pinned in requirements.txt).
```bash
conda create -n polystruct-mine python=3.10 -y
conda activate polystruct-mine
pip install -r requirements.txt
```
Key deps: rdkit-pypi (or conda-forge rdkit), jpype1, streamlit, pandas, pyarrow.

Java/OPSIN requirements
- Java JDK/JRE 17+ recommended. Verify: `java -version`.
- OPSIN JAR: bundled at `OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar` (prebuilt). If missing, build:
  ```bash
  cd OPSIN
  mvn -pl opsin-cli -am package -DskipTests
  ```
  Maven will create the shaded JAR under `opsin-cli/target/`.
- JPype bridge: `polymer_pipeline/utils/opsin_bridge.py` starts JVM with:
  - JAR path: `/Users/k25063738/Text-mining-postprocessing/OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar` (update to your local path if needed)
  - JVM library path: `JVM_PATH = "/opt/homebrew/Cellar/openjdk/25.0.1/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"` (macOS arm64 example; adjust per OS)
  - Cache: `~/.opsin/cache.json` (persisted SMILES per name; flush by deleting the file). Logging uses `polymer_pipeline` logger.
- Important: JVM must start only once per process; restart Streamlit after changing JVM args.

Java / JVM Installation (Required for OPSIN)
--------------------------------------------
OPSIN is a Java library; JPype loads the OPSIN CLI JAR. Use an LTS JDK (17+ recommended).

- macOS (Homebrew):
  ```bash
  brew install --cask temurin@17
  java -version
  /usr/libexec/java_home -V
  export JAVA_HOME=$(/usr/libexec/java_home -v 17)
  ```
  Optional Maven (for rebuilding OPSIN): `brew install maven` then `mvn -v`.

- Ubuntu/Debian:
  ```bash
  sudo apt update
  sudo apt install -y openjdk-17-jdk maven
  java -version
  mvn -v
  # Optional JAVA_HOME
  export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
  ```

- Windows/WSL2:
  - Install Temurin/OpenJDK 17 via installer or winget; on WSL use `sudo apt install -y openjdk-17-jdk`.
  - Verify: `java -version`. Set `JAVA_HOME` to the JDK root if needed.

OPSIN build (if jar missing)
```bash
cd OPSIN
mvn -pl opsin-cli -am package -DskipTests
ls opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar
```

JPype/JVM troubleshooting:
- `java: command not found`: install JDK and ensure PATH/JAVA_HOME set.
- JAR not found: point `OPSIN_JAR_PATH` in `polymer_pipeline/utils/opsin_bridge.py` to the built JAR.
- JVM cannot start: verify `JVM_PATH` (e.g., `python - <<'PY'\nimport jpype; print(jpype.getDefaultJVMPath())\nPY`), restart process if JVM already started.
- Clear OPSIN cache: remove `~/.opsin/cache.json`.

Running the Streamlit UI
------------------------
```bash
streamlit run ui_app.py
```
Features:
- Upload/select text polymer CSV and image pool CSV.
- Run pipeline (cached by file signature); outputs `polymer_structures_groundtruth.csv` and `logs/statistics_report.json`.
- Preview results table; download CSV/LLM export.
- SMILES Scratchpad Tester (isolated) to repair/canonicalize arbitrary SMILES and render RDKit SVG (does not affect pipeline).
- SMILES Scratchpad Tester (isolated) to repair/canonicalize arbitrary SMILES and render RDKit SVG (does not affect pipeline).
Logs: `logs/run_summary.log`; metrics: `logs/statistics_report.json`; OPSIN traces: `logs/opsin_trace.csv`; candidate scores: `logs/candidate_scores.csv`.

Running pipeline via CLI
------------------------
```bash
python -m polymer_pipeline.pipeline \
  --text_table data/example_polymer_data.csv \
  --image_pool data/example_image_candidates.csv \
  --output outputs/polymer_structures_groundtruth.csv \
  --fast_mode False
```
Outputs to `polymer_structures_groundtruth.csv`; logs under `logs/`.

OPSIN smoke test
----------------
```bash
python - <<'PY'
from polymer_pipeline.utils.opsin_bridge import parse_chemical_name, get_opsin_version
print("OPSIN version:", get_opsin_version())
print("ethanol ->", parse_chemical_name("ethanol"))
PY
```
If None is printed, check JVM/JAR paths and Java install.

RDKit smoke test (draw SVG)
---------------------------
```bash
python - <<'PY'
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
mol = Chem.MolFromSmiles("c1ccccc1")
drawer = rdMolDraw2D.MolDraw2DSVG(200,200)
drawer.DrawMolecule(mol); drawer.FinishDrawing()
svg = drawer.GetDrawingText()
print("SVG len:", len(svg))
PY
```

End-to-end sample run
---------------------
```bash
python -m polymer_pipeline.pipeline \
  --text_table examples/text_polymers_sample.csv \
  --image_pool examples/image_pool_sample.csv \
  --output outputs/polymer_structures_groundtruth.csv
```
Check outputs and logs in `outputs/` and `logs/`.

Troubleshooting
---------------
- OPSIN jar not found / JVM fails: update `OPSIN_JAR_PATH` and `JVM_PATH` in `polymer_pipeline/utils/opsin_bridge.py`; ensure Java installed; restart process.
- JPype ImportError: install `jpype1` (matching Python arch); restart.
- RDKit missing: install via conda-forge (`conda install -c conda-forge rdkit`) or pip wheel if compatible.
- Streamlit SVG blank: ensure RDKit imported; tester uses `st.image` with SVG.
- Performance: enable `fast_mode` to skip some retries; caches: OPSIN cache under `~/.opsin/cache.json`, pipeline cache via Streamlit session and `_RD_PARSE_CACHE` in Layer 2.

Reproducibility and auditability
--------------------------------
- Deterministic, rule-based; no external API calls.
- Schema report: `logs/schema_report.json`.
- OPSIN trace: `logs/opsin_trace.csv`; candidate scores: `logs/candidate_scores.csv`; repair log: `logs/repair_log.csv`.
- Metrics: `logs/statistics_report.json`; coverage/confidence in console logs.

License and citation
--------------------
- MIT License (if present in repo).
- Cite: “PolyStruct-Mine: Deterministic fusion of text- and image-mined polymer repeat units for structure–property datasets, 2025.” (replace with DOI/arXiv when available).
