import logging
import re
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from polymer_pipeline.schema_utils import (
    normalize_image_columns,
    normalize_polymer_columns,
    write_schema_report,
)
from polymer_pipeline.utils import (
    normalize_confidence,
    normalize_polymer_name,
    normalize_sidechain_name,
    sanitize_smiles,
    safe_apply_smiles,
)

logger = logging.getLogger("polymer_pipeline")


def validate_and_normalize(text_df: pd.DataFrame, image_df: pd.DataFrame, schema_report_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Layer 0: schema-driven normalization. Maps variant columns to canonical names, fills missing columns,
    normalizes text fields, sanitizes SMILES, and writes schema report.
    """

    paper_key_candidates = ["paper_key", "paper_key_display", "_source_file", "source_file", "file_name"]

    def _parse_numeric_prefix(val):
        if not isinstance(val, str):
            return None
        m = re.match(r"^([0-9]+)", val.strip())
        return int(m.group(1)) if m else None

    def _detect_paper_key_column(df: pd.DataFrame) -> Optional[str]:
        for col in paper_key_candidates:
            if col in df.columns:
                return col
        return None

    def _attach_paper_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Centralized paper id extraction:
        - Detect a key-like column (paper_key/_source_file/source_file/file_name).
        - Parse numeric prefix once and use for paper_id_display and paper_id (when missing/empty).
        - If paper_name exists but no key column, parse numeric prefix from paper_name as a fallback.
        - Preserve full key in paper_key_display when available.
        """
        df = df.copy()
        key_col = _detect_paper_key_column(df)
        if key_col is not None:
            parsed = df[key_col].apply(_parse_numeric_prefix)
        elif "paper_name" in df.columns:
            parsed = df["paper_name"].apply(_parse_numeric_prefix)
        else:
            parsed = pd.Series([None] * len(df), index=df.index)

        if key_col and "paper_key_display" not in df.columns:
            df["paper_key_display"] = df[key_col]

        has_display = "paper_id_display" in df.columns and df["paper_id_display"].notna().any()
        if not has_display:
            df["paper_id_display"] = parsed

        if "paper_id" not in df.columns:
            df["paper_id"] = parsed
        else:
            mask_missing = df["paper_id"].isna() | (df["paper_id"].astype(str).str.strip() == "")
            df.loc[mask_missing, "paper_id"] = parsed[mask_missing]
        return df

    def ensure_dataframe(obj, name: str) -> pd.DataFrame:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.Series):
            obj = obj.to_frame().T if obj.name else obj.to_frame()
        elif isinstance(obj, (list, dict)):
            obj = pd.DataFrame(obj)
        elif not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception:
                return pd.DataFrame()
        obj = obj.copy()
        obj.reset_index(drop=True, inplace=True)
        obj.columns = [str(c) for c in obj.columns]
        return obj

    text_df = ensure_dataframe(text_df, "text_df")
    image_df = ensure_dataframe(image_df, "image_df")
    # Preserve existing backbone/side-chain columns by copying into canonical names instead of overwriting with None.
    if "backbone_name_text" not in text_df.columns and "backbone" in text_df.columns:
        text_df["backbone_name_text"] = text_df["backbone"]
    if "sidechain_name_text" not in text_df.columns and "side_chain" in text_df.columns:
        text_df["sidechain_name_text"] = text_df["side_chain"]
    for col in ["paper_id", "polymer_name", "backbone_name_text", "sidechain_name_text", "confidence_text"]:
        if col not in text_df.columns:
            text_df[col] = None
    for col in ["paper_id", "candidate_smiles"]:
        if col not in image_df.columns:
            image_df[col] = None
    text_norm, text_map, text_missing = normalize_polymer_columns(text_df)
    image_norm, image_map, image_missing = normalize_image_columns(image_df)
    assert isinstance(text_norm, pd.DataFrame), "text_norm must be a DataFrame"
    assert isinstance(image_norm, pd.DataFrame), "image_norm must be a DataFrame"
    logger.info("Layer 0 input columns (polymer): %s", list(text_df.columns))
    logger.info("Layer 0 input columns (image): %s", list(image_df.columns))
    logger.info("Layer 0 normalized columns (polymer): %s", list(text_norm.columns))
    logger.info("Layer 0 normalized columns (image): %s", list(image_norm.columns))

    text_norm = _attach_paper_ids(text_norm)
    image_norm = _attach_paper_ids(image_norm)

    def _normalized_name(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower()

    # Unified paper_id mapping across text and image by normalized paper_name.
    text_names = _normalized_name(text_norm["paper_name"]) if "paper_name" in text_norm.columns else pd.Series([], dtype=str)
    image_names = _normalized_name(image_norm["paper_name"]) if "paper_name" in image_norm.columns else pd.Series([], dtype=str)
    all_names = pd.concat([text_names, image_names], ignore_index=True)
    mapping = {}
    if not all_names.empty:
        unique_names = [n for n in pd.unique(all_names) if n]
        mapping = {name: idx + 1 for idx, name in enumerate(unique_names)}

    def _assign_paper_id(df: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
        """
        Ensure paper_id is populated deterministically using existing ids when present,
        else derive from shared mapping of normalized paper_name,
        else fall back to row index.
        """
        df = df.copy()
        if "paper_id" in df.columns:
            df["paper_id"] = df["paper_id"].fillna("").astype(str).str.strip()
            if df["paper_id"].str.len().sum() > 0:
                return df

        mapped = names.map(mapping) if not names.empty else pd.Series(pd.NA, index=df.index)
        if mapped.notna().any():
            df["paper_id"] = mapped.fillna(pd.Series(range(1, len(df) + 1), index=df.index)).astype(int)
            return df

        df["paper_id"] = df.index + 1
        return df

    text_norm = _assign_paper_id(text_norm, text_names if not text_names.empty else pd.Series([pd.NA] * len(text_norm)))
    image_norm = _assign_paper_id(image_norm, image_names if not image_names.empty else pd.Series([pd.NA] * len(image_norm)))

    # Text normalization
    if "polymer_name" in text_norm.columns:
        text_norm["polymer_name"] = (
            text_norm["polymer_name"].fillna("").astype(str).str.strip().str.lower()
        )
        text_norm["polymer_name_norm"] = text_norm["polymer_name"]
    if "paper_id" in text_norm.columns:
        text_norm["paper_id"] = text_norm["paper_id"].astype(str).str.strip().str.lower()
    if "backbone_name_text" in text_norm.columns:
        text_norm["backbone_name_text"] = text_norm["backbone_name_text"].fillna("").apply(normalize_polymer_name)
    if "sidechain_name_text" in text_norm.columns:
        text_norm["sidechain_name_text"] = text_norm["sidechain_name_text"].fillna("").apply(normalize_sidechain_name)
    if "confidence_text" in text_norm.columns:
        text_norm["confidence_text_norm"] = text_norm["confidence_text"].apply(normalize_confidence)

    # Image normalization
    if "candidate_smiles" in image_norm.columns:
        image_norm = safe_apply_smiles(image_norm, "candidate_smiles")
    elif "canonical_smiles" in image_norm.columns:
        image_norm = safe_apply_smiles(image_norm, "canonical_smiles")
        image_norm["candidate_smiles_clean"] = image_norm.get("canonical_smiles_clean")
    elif "raw_smiles" in image_norm.columns:
        image_norm = safe_apply_smiles(image_norm, "raw_smiles")
        image_norm["candidate_smiles_clean"] = image_norm.get("raw_smiles_clean")
    else:
        image_norm["candidate_smiles_clean"] = None
    if "paper_id" in image_norm.columns:
        image_norm["paper_id"] = image_norm["paper_id"].fillna("").astype(str).str.strip().str.lower()
    else:
        image_norm["paper_id"] = image_norm.index.astype(str)
    if "polymer_name" in image_norm.columns:
        image_norm["polymer_name"] = image_norm["polymer_name"].astype(str).str.strip().str.lower()

    mappings = {"polymer": text_map, "image": image_map}
    missing = {"polymer": text_missing, "image": image_missing}
    if schema_report_path:
        write_schema_report(text_norm, image_norm, schema_report_path, mappings=mappings, missing=missing)

    logger.info("Layer 0: normalized schemas (missing polymer cols=%s, missing image cols=%s)", text_missing, image_missing)
    logger.info(
        "Layer 0: unique paper_ids (polymer=%d, image=%d)",
        text_norm["paper_id"].nunique(),
        image_norm["paper_id"].nunique(),
    )
    logger.info("Layer 0 backbone preview: %s", list(text_norm.get("backbone_name_text", pd.Series()).head(5)))
    logger.info("Layer 0 sidechain preview: %s", list(text_norm.get("sidechain_name_text", pd.Series()).head(5)))
    return text_norm, image_norm
