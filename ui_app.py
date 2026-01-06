"""
Streamlit dashboard for PolyStruct-Mine.

Launch with:
    streamlit run ui_app.py
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple
import ast

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception:
    Chem = None
    Draw = None
    rdDepictor = None
    rdMolDraw2D = None

from polymer_pipeline.logging_utils import setup_logger
from polymer_pipeline.metrics import compute_metrics
from polymer_pipeline.pipeline import run_pipeline
from polymer_pipeline.export_llm import export_llm_ready
from polymer_pipeline.layers.layer2_fusion import (
    preclean_smiles,
    validate_and_canonicalize,
    auto_correct_smiles,
)
from polymer_pipeline.utils import rdkit_tools

LOG_FILE = "logs/run_summary.log"
DEFAULT_OUTPUT = "polymer_structures_groundtruth.csv"
LAST_UPLOAD = Path("data/last_upload.json")
USER_POLY = Path("data/user_polymer.csv")
USER_IMAGE = Path("data/user_image_pool.csv")


def _render_smiles_image(smiles):
    if not smiles or Chem is None or rdMolDraw2D is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if rdDepictor is not None:
            rdDepictor.SetPreferCoordGen(True)
        drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace("svg:", "").replace('xmlns:svg=', 'xmlns=')
        return svg
    except Exception:
        return None


def _normalize_ids_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a robust, human-readable paper_id_display column for the UI."""
    if df is None or df.empty:
        return df
    df = df.copy()

    key_candidates = ["paper_key", "paper_key_display", "_source_file", "source_file", "file_name"]

    def _parse_numeric_prefix(val):
        if not isinstance(val, str):
            return None
        m = re.match(r"^(\d+)", val.strip())
        return int(m.group(1)) if m else None

    def _detect_key_column(df_local: pd.DataFrame):
        for col in key_candidates:
            if col in df_local.columns:
                return col
        return None

    def _resolve_paper_id_display(df_local: pd.DataFrame) -> pd.DataFrame:
        if "paper_id_display" in df_local.columns and df_local["paper_id_display"].notna().any():
            return df_local

        key_col = _detect_key_column(df_local)
        parsed = None
        if key_col:
            parsed = df_local[key_col].apply(_parse_numeric_prefix)
            df_local["paper_id_display"] = parsed
        elif "paper_id" in df_local.columns:
            df_local["paper_id_display"] = df_local["paper_id"]
        else:
            df_local["paper_id_display"] = [i + 1 for i in range(len(df_local))]

        if "paper_id" in df_local.columns:
            mask_missing = df_local["paper_id"].isna() | (df_local["paper_id"].astype(str).str.strip() == "")
            if parsed is not None:
                df_local.loc[mask_missing, "paper_id"] = parsed[mask_missing]
        return df_local

    id_candidates = [c for c in df.columns if c.lower() in ["paper_id", "paper", "paperid", "file_id", "document_id"]]
    if id_candidates:
        df.rename(columns={id_candidates[0]: "paper_id"}, inplace=True)

    df = _resolve_paper_id_display(df)
    return df


def _file_sig(path: str) -> str:
    p = Path(path)
    try:
        stat = p.stat()
        return f"{p.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    except FileNotFoundError:
        return str(p)


@st.cache_data(show_spinner=False)
def _run_pipeline_cached(text_path: str, image_path: str, verbose: bool, fast_mode: bool, sig_text: str, sig_image: str):
    """Cache full pipeline run by input file signature to avoid reruns on UI interactions."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logger(log_file=LOG_FILE, level=level)
    df = run_pipeline(text_path, image_path, output_path=DEFAULT_OUTPUT, fast_mode=fast_mode)
    metrics_raw = compute_metrics(df)
    metrics = {}
    for k, v in metrics_raw.items():
        if isinstance(v, dict):
            metrics[k] = {kk: int(vv) for kk, vv in v.items()}
        else:
            metrics[k] = float(v)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("logs/statistics_report.json").write_text(json.dumps(metrics, indent=2))
    return df, metrics, sig_text, sig_image


def _render_review(df: pd.DataFrame, paper_query: str = ""):
    df = _normalize_ids_for_display(df)
    if not paper_query or "paper_id" not in df.columns:
        return

    df_filtered = df[df["paper_id"].astype(str).str.contains(paper_query.strip(), case=False, na=False)]

    if df_filtered.empty:
        st.warning("No entries found for this Paper ID.")
        return

    st.success(f"Found {len(df_filtered)} polymer entries for Paper ID: {paper_query}")
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore
    except Exception:
        Chem = None
        rdMolDraw2D = None

    def render_smiles(smiles):
        if Chem is None or rdMolDraw2D is None:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if rdDepictor is not None:
                rdDepictor.SetPreferCoordGen(True)
            if not mol:
                return None
            drawer = rdMolDraw2D.MolDraw2DSVG(220, 220)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg = svg.replace("svg:", "").replace('xmlns:svg=', 'xmlns=')
            return svg
        except Exception:
            return None

    st.markdown(f"### Molecule Structures for Paper ID: {paper_query}")
    for _, row in df_filtered.iterrows():
        st.markdown(f"**{row.get('polymer_name','')} ({row.get('paper_id_display', row.get('paper_id',''))})**")
        cols = st.columns(3)
        with cols[0]:
            img = render_smiles(row.get("backbone_smiles"))
            if img:
                st.image(img, caption="Backbone")
            else:
                st.write("Backbone: N/A")
        with cols[1]:
            sc = row.get("sidechain_smiles_list")
            if isinstance(sc, (list, tuple)):
                for s in sc:
                    img = render_smiles(s)
                    if img:
                        st.image(img, caption="Side Chain")
            else:
                img = render_smiles(sc)
                if img:
                    st.image(img, caption="Side Chain")
                else:
                    st.write("Side Chain: N/A")
        with cols[2]:
            img = render_smiles(row.get("repeat_unit_smiles_full"))
            if img:
                st.image(img, caption="Repeat Unit")
            else:
                st.write("Repeat Unit: N/A")
        st.markdown("---")


def _render_svg_from_mol(mol, width: int = 220, height: int = 220):
    if mol is None or rdMolDraw2D is None:
        return None
    try:
        if rdDepictor is not None:
            rdDepictor.SetPreferCoordGen(True)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace("svg:", "").replace('xmlns:svg=', 'xmlns=')
        return svg
    except Exception:
        return None


def _smiles_test_pipeline(smiles: str, use_pipeline_repair: bool = True):
    """
    Isolated tester: does not affect pipeline state. Returns diagnostic dict.
    """
    result = {
        "input": smiles,
        "precleaned": None,
        "canonical": None,
        "fragment_count_in": smiles.count(".") if isinstance(smiles, str) else 0,
        "rgroup_replaced": False,
        "fragments_dropped": False,
        "skip_reason": None,
        "rdkit_error": None,
    }
    if not isinstance(smiles, str) or not smiles.strip():
        result["skip_reason"] = "empty"
        return result
    try:
        pre = preclean_smiles(smiles)
        result["precleaned"] = pre
        if pre is None:
            result["skip_reason"] = "preclean_failed_or_overlong"
            return result
        result["rgroup_replaced"] = "[R" in smiles and "[*]" in pre
        if smiles.count(".") > 0 and pre.count(".") == 0 and pre != smiles:
            result["fragments_dropped"] = True
        if use_pipeline_repair:
            canonical = auto_correct_smiles(pre)
        else:
            if rdkit_tools.Chem is None:
                canonical = pre
            else:
                mol = rdkit_tools.Chem.MolFromSmiles(pre, sanitize=True)
                canonical = rdkit_tools.Chem.MolToSmiles(mol, canonical=True) if mol else None
        result["canonical"] = canonical
    except Exception as exc:
        result["rdkit_error"] = str(exc)
    return result


def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _coerce_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely coerce known list-like columns when reloaded from CSV/XLSX.
    """

    def _coerce_cell_to_list_or_scalar(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        if x is None:
            return []
        try:
            if pd.isna(x):
                return []
        except Exception:
            pass
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    v = ast.literal_eval(s)
                    return list(v) if isinstance(v, (list, tuple)) else v
                except Exception:
                    return x
        return x

    list_cols = [
        "sidechain_smiles_list",
        "sidechain_smiles_list_text",
        "sidechain_smiles_list_text_maybe",
    ]
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_coerce_cell_to_list_or_scalar)
    return df


def _load_corrected_table(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    if uploaded_file is None:
        return None, "No file uploaded"
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type"
        df.columns = [c.strip() for c in df.columns]

        # Column aliasing for corrected tables
        if "paper_id" not in df.columns and "paper_id_display" in df.columns:
            df["paper_id"] = df["paper_id_display"]
        if "backbone_smiles_text" not in df.columns and "backbone_smiles" in df.columns:
            df["backbone_smiles_text"] = df["backbone_smiles"]

        # Build sidechain_smiles_list/text from numbered columns if present
        numbered_cols = [c for c in ["sidechain_smiles_text_1", "sidechain_smiles_text_2", "sidechain_smiles_text_3"] if c in df.columns]
        if numbered_cols:
            def _assemble_sidechains(row):
                vals = []
                for col in numbered_cols:
                    val = row[col] if col in row else None
                    if isinstance(val, str) and val.strip():
                        vals.append(val.strip())
                return vals
            df["sidechain_smiles_list"] = df.apply(_assemble_sidechains, axis=1)
            df["sidechain_smiles_text"] = df["sidechain_smiles_list"].apply(lambda lst: lst[0] if isinstance(lst, list) and len(lst) > 0 else None)
            df["sidechain_smiles_list_text"] = df["sidechain_smiles_list"].apply(lambda lst: str(lst) if isinstance(lst, list) else None)

        df = _coerce_list_columns(df)
        required = ["paper_id"]
        missing = [r for r in required if r not in df.columns]
        if missing:
            return None, f"Missing required columns: {missing}"
        return df, None
    except Exception as exc:
        return None, str(exc)


def render_review_panel(df: pd.DataFrame, panel_key: str = "main"):
    """
    Reusable review panel; keys are namespaced by panel_key to avoid collisions.
    """
    if df is None or df.empty:
        st.info("No data to display.")
        return
    df_local = _normalize_ids_for_display(df)
    review_cols = [
        "paper_id_display",
        "polymer_name",
        "backbone_name_text",
        "sidechain_name_text",
        "sidechain_smiles_text",
        "repeat_unit_smiles_full",
        "backbone_smiles",
        "confidence",
        "source_priority",
        "needs_manual_review",
    ]
    visible = [c for c in review_cols if c in df_local.columns]
    if visible:
        st.dataframe(df_local[visible])
    st.markdown("### Search by Paper ID")
    paper_query = st.text_input(
        "Enter a Paper ID to review molecule structures",
        "",
        key=f"paper_id_query_{panel_key}",
    )
    _render_review(df_local, paper_query=paper_query)


def _persist_upload(uploaded_file, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded_file.getbuffer())
    uploads = {"polymer": str(USER_POLY) if USER_POLY.exists() else None, "image_pool": str(USER_IMAGE) if USER_IMAGE.exists() else None}
    LAST_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    LAST_UPLOAD.write_text(json.dumps(uploads, indent=2))


def _run_pipeline(text_path: str, image_path: str, verbose: bool) -> Tuple[pd.DataFrame, dict]:
    level = logging.DEBUG if verbose else logging.INFO
    setup_logger(log_file=LOG_FILE, level=level)
    df = run_pipeline(text_path, image_path, output_path=DEFAULT_OUTPUT)
    metrics_raw = compute_metrics(df)
    metrics = {}
    for k, v in metrics_raw.items():
        if isinstance(v, dict):
            metrics[k] = {kk: int(vv) for kk, vv in v.items()}
        else:
            metrics[k] = float(v)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("logs/statistics_report.json").write_text(json.dumps(metrics, indent=2))
    return df, metrics


def _render_metrics(metrics: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Coverage (side-chain)", f"{metrics['coverage_sidechain']*100:.1f}%")
    col2.metric("Coverage (backbone)", f"{metrics['coverage_backbone']*100:.1f}%")
    col3.metric("Coverage (repeat-unit)", f"{metrics['coverage_repeat_unit']*100:.1f}%")
    st.metric("Manual review rate", f"{metrics['manual_review_rate']*100:.1f}%")
    if metrics.get("top_failure_reasons"):
        reasons = pd.DataFrame(
            {"reason": list(metrics["top_failure_reasons"].keys()), "count": list(metrics["top_failure_reasons"].values())}
        )
        fig = px.bar(reasons, x="reason", y="count", title="Top failure reasons")
        st.plotly_chart(fig, use_container_width=True)


def _render_opsin_status_panel():
    """
    Lightweight, self-contained OPSIN diagnostics for the sidebar.
    Never raises: all checks are wrapped in try/except to stay non-intrusive.
    """
    jpype_status = "❌ JPype not importable"
    jvm_status = "❌ JVM path not found"
    jar_status = "❌ OPSIN jar missing"
    version_status = "❌ Version unknown"
    package_status = "❌ Package unknown"
    parse_status = "❌ Not attempted"
    hint_lines = []

    opsin_bridge = None
    jvm_status_data = {"started": False, "error": None, "jvm_path": None}
    try:
        from polymer_pipeline.utils import opsin_bridge  # type: ignore

        jar_path = getattr(
            opsin_bridge,
            "OPSIN_JAR_PATH",
            Path("/Users/k25063738/Text-mining-postprocessing/OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar"),
        )
        jvm_status_data = opsin_bridge.get_jvm_status()
    except Exception:
        jar_path = Path("/Users/k25063738/Text-mining-postprocessing/OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar")

    try:
        import jpype  # type: ignore

        jpype_status = f"✅ JPype {getattr(jpype, '__version__', '')}".strip()
        try:
            jvm_path = jpype.getDefaultJVMPath()
            if jvm_path:
                jvm_status = f"✅ JVM path: {jvm_path}"
            else:
                jvm_status = "❌ JVM path not resolved"
                hint_lines.append("JPype could not resolve the JVM path.")
        except Exception as exc:
            jvm_status = f"❌ JVM path error: {exc}"
            hint_lines.append("JPype failed to locate the JVM; ensure JAVA_HOME is set.")
    except Exception as exc:
        hint_lines.append("Install jpype1 in the active environment.")
        jpype_status = f"❌ JPype import failed: {exc}"

    try:
        if jar_path.exists():
            jar_status = f"✅ OPSIN jar at {jar_path}"
        else:
            jar_status = "❌ OPSIN jar missing"
            hint_lines.append("OPSIN jar missing; build or place the 3.0-SNAPSHOT jar.")
    except Exception:
        pass

    try:
        from polymer_pipeline.utils import opsin_bridge  # type: ignore
        version_status = f"✅ Version {opsin_bridge.get_opsin_version()}"
        if getattr(opsin_bridge, "OPSIN_PACKAGE", None):
            package_status = f"✅ OPSIN Java backend: {opsin_bridge.OPSIN_PACKAGE}"
        else:
            package_status = "❌ OPSIN Java class missing — please verify the jar includes NameToStructure"
        if not jvm_status_data.get("started"):
            error_text = jvm_status_data.get("error")
            jvm_status = "❌ JVM failed to start"
            hint_lines.append("JVM failed to start; please check Java version.")
            st.sidebar.warning("❌ JVM failed to start. Please remove unsupported JVM arguments or use Java 21.")
        ethanol = opsin_bridge.parse_chemical_name("ethanol")
        if ethanol:
            parse_status = f"✅ Parsed ethanol → {ethanol}"
        else:
            parse_status = "❌ Parsing returned no SMILES"
            hint_lines.append("OPSIN bridge returned no result for a simple name.")
        if not getattr(opsin_bridge, "OPSIN_AVAILABLE", False):
            hint_lines.append("OPSIN_AVAILABLE is False inside opsin_bridge.")
    except Exception as exc:
        parse_status = f"❌ Parsing test failed: {exc}"
        hint_lines.append("OPSIN bridge import or call failed.")

    with st.sidebar.expander("OPSIN System Status", expanded=False):
        st.write(jpype_status)
        st.write(jvm_status)
        st.write(jar_status)
        st.write(version_status)
        st.write(package_status)
        st.write(parse_status)
        st.write(f"Python executable: {sys.executable}")
        if hint_lines:
            st.caption("Hints: " + " | ".join(dict.fromkeys(hint_lines)))


def main():
    st.set_page_config(page_title="PolyStruct-Mine", layout="wide")
    st.title("PolyStruct-Mine — Polymer Structure Mining Pipeline")

    st.sidebar.header("Molecule Review")
    _render_opsin_status_panel()
    verbose = st.sidebar.checkbox("Verbose logging", value=False)
    fast_mode = st.sidebar.checkbox("Fast mode (skip retries)", value=False)
    polymer_upload = st.sidebar.file_uploader("Polymer/text dataset", type=["csv", "json", "parquet"])
    image_upload = st.sidebar.file_uploader("Image/candidate dataset", type=["csv", "json", "parquet"])
    run_clicked = st.sidebar.button("Run Pipeline")

    # Reuse last uploads if available and no new upload provided.
    polymer_path = USER_POLY if USER_POLY.exists() else None
    image_path = USER_IMAGE if USER_IMAGE.exists() else None
    if polymer_upload:
        _persist_upload(polymer_upload, USER_POLY)
        polymer_path = USER_POLY
    if image_upload:
        _persist_upload(image_upload, USER_IMAGE)
        image_path = USER_IMAGE

    output_placeholder = st.empty()
    metrics_placeholder = st.empty()

    paper_query = ""
    df_current = None

    if run_clicked:
        if not polymer_path or not image_path:
            st.error("Please upload both polymer and image pool files.")
        else:
            with st.spinner("Running pipeline..."):
                sig_text = _file_sig(str(polymer_path))
                sig_image = _file_sig(str(image_path))
                df, metrics, _, _ = _run_pipeline_cached(str(polymer_path), str(image_path), verbose, fast_mode, sig_text, sig_image)
            st.success("Pipeline completed.")
            metrics_placeholder.write("### Metrics")
            _render_metrics(metrics)
            output_placeholder.write("### Results preview")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv_bytes, file_name=DEFAULT_OUTPUT, mime="text/csv")
            df_current = df
    else:
        out_path = Path(DEFAULT_OUTPUT)
        metrics_path = Path("logs/statistics_report.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            metrics_placeholder.write("### Metrics (latest)")
            _render_metrics(metrics)
        if out_path.exists():
            df = pd.read_csv(out_path)
            if "sidechain_smiles_list" in df.columns:
                df["sidechain_smiles_list"] = df["sidechain_smiles_list"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
                )
            output_placeholder.write("### Results preview (latest)")
            st.download_button("Download results CSV", data=df.to_csv(index=False), file_name=DEFAULT_OUTPUT, mime="text/csv")
            df_current = df

    st.markdown("---")
    with st.expander("SMILES Scratchpad Tester (does not affect table)", expanded=False):
        st.caption("Isolated tester; does not modify pipeline outputs or cache.")
        tester_input = st.text_area("Enter SMILES to test", value=st.session_state.get("smiles_tester_input", ""), height=120)
        use_repair = st.checkbox("Use pipeline repair (Layer2 auto_correct_smiles)", value=True)
        col_a, col_b = st.columns(2)
        test_clicked = col_a.button("Test & Render")
        clear_clicked = col_b.button("Clear")
        if clear_clicked:
            st.session_state["smiles_tester_input"] = ""
            st.session_state["smiles_tester_last_result"] = None
            _safe_rerun()
        if test_clicked:
            st.session_state["smiles_tester_input"] = tester_input
            res = _smiles_test_pipeline(tester_input, use_pipeline_repair=use_repair)
            st.session_state["smiles_tester_last_result"] = res
        res = st.session_state.get("smiles_tester_last_result")
        if res:
            st.markdown("**Result**")
            st.text(f"Input: {res.get('input')}")
            st.text(f"Precleaned: {res.get('precleaned')}")
            st.text(f"Canonical: {res.get('canonical')}")
            st.text(f"Fragments in input: {res.get('fragment_count_in')}")
            st.text(f"R-group replaced: {res.get('rgroup_replaced')}")
            st.text(f"Fragments dropped: {res.get('fragments_dropped')}")
            if res.get("skip_reason"):
                st.warning(f"Skip: {res.get('skip_reason')}")
            if res.get("rdkit_error"):
                st.error("RDKit error")
                st.code(res.get("rdkit_error"))
            svg = None
            if res.get("canonical") and rdkit_tools.Chem is not None:
                try:
                    mol = rdkit_tools.Chem.MolFromSmiles(res["canonical"])
                    svg = _render_svg_from_mol(mol)
                except Exception as exc:
                    st.error("Render failed")
                    st.code(str(exc))
            if svg:
                st.image(svg, caption="SMILES Tester", use_container_width=False)
            elif res.get("canonical"):
                st.info("Canonical SMILES available but rendering failed.")
            debug_on = st.checkbox("Advanced debug", value=False, key="smiles_tester_debug_on")
            if debug_on:
                st.code((svg or "")[:300] if svg else "No SVG", language="xml")

    if df_current is not None:
        render_review_panel(df_current, panel_key="main")
        st.markdown("### LLM Review Export")
        if st.button("Download LLM-friendly JSON", key="generate_llm_export"):
            export_path = Path("exports/llm_ready.json")
            try:
                export_llm_ready(df_current, export_path)
                st.success(f"LLM-friendly file generated: {export_path}")
                with open(export_path, "rb") as fh:
                    st.download_button(
                        label="Download LLM Review File",
                        data=fh.read(),
                        file_name=export_path.name,
                        mime="application/json",
                    )
            except Exception as exc:
                st.warning(f"Export failed: {exc}")

    st.caption("PolyStruct-Mine · Deterministic polymer structure mining · MIT License · 2025")

    # View-only corrected table uploader and viewer (no pipeline run, isolated state)
    st.divider()
    st.info("Upload corrected table (CSV/XLSX) for review only — no pipeline run will occur.")
    uploaded_corrected = st.file_uploader(
        "Upload corrected table (CSV / XLSX)",
        type=["csv", "xlsx"],
        key="corrected_table_uploader",
    )
    corrected_df = None
    if uploaded_corrected:
        corrected_df, err = _load_corrected_table(uploaded_corrected)
        if err:
            st.error(f"Could not load corrected table: {err}")
        else:
            st.success(f"Corrected table loaded: {len(corrected_df)} rows, {len(corrected_df.columns)} columns")
            render_review_panel(corrected_df, panel_key="corrected")


if __name__ == "__main__":  # pragma: no cover
    main()
