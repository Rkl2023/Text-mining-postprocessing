import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from polymer_pipeline.utils import safe_json_dumps
logger = logging.getLogger("polymer_pipeline")

# Canonical field names and their possible variants in real data.
POLYMER_FIELD_ALIASES: Dict[str, List[str]] = {
    "paper_id": ["paper_id", "paperid", "paper", "_source_file"],
    "paper_name": ["paper_name", "paper_title"],
    "polymer_name": ["polymer_name", "polymer", "polymerid"],
    "backbone_name_text": ["backbone_name_text", "backbone", "backbone_text"],
    "sidechain_name_text": ["sidechain_name_text", "side_chain", "side_chain_text", "sidechain_desc"],
    "device_type": ["device_type"],
    "confidence_text": ["confidence_text", "confidence"],
}

IMAGE_FIELD_ALIASES: Dict[str, List[str]] = {
    "paper_id": ["paper_id", "paperid", "paper"],
    "paper_name": ["paper_name", "paper_title"],
    "polymer_name": ["polymer_name", "polymer", "polymerid"],
    "image_id": ["image_id", "image", "image_path"],
    "fragment_id": ["fragment_id", "segment_index"],
    "candidate_smiles": ["candidate_smiles", "cleaned_smiles", "canonical_smiles", "raw_smiles"],
    "parse_status": ["parse_status", "status"],
}


def _normalize_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Return df with canonical columns added (if found), mapping source->canonical, and missing canonical fields."""
    df = df.copy()
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    lower_cols = {c.lower(): c for c in df.columns}

    for canonical, variants in aliases.items():
        found_source = None
        for variant in variants:
            if variant.lower() in lower_cols:
                found_source = lower_cols[variant.lower()]
                break
        if found_source:
            df[canonical] = df[found_source]
            mapping[found_source] = canonical
        else:
            df[canonical] = None
            missing.append(canonical)
    return df, mapping, missing


def normalize_polymer_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    return _normalize_columns(df, POLYMER_FIELD_ALIASES)


def normalize_image_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    return _normalize_columns(df, IMAGE_FIELD_ALIASES)


def infer_schema(df: pd.DataFrame) -> Dict:
    cols = list(df.columns)
    nullable = [col for col in cols if df[col].isna().any()]
    examples = {}
    if not df.empty:
        sample = df.iloc[0]
        for col in cols:
            examples[col] = sample[col] if pd.notna(sample[col]) else None
    return {"columns": cols, "nullable": nullable, "examples": examples}


def write_schema_report(poly_df: pd.DataFrame, img_df: pd.DataFrame, report_path: Path, mappings: Dict, missing: Dict):
    report = {
        "polymer_table": infer_schema(poly_df),
        "image_pool": infer_schema(img_df),
        "mappings": mappings,
        "missing": missing,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    safe_json_dumps(report, path=report_path)
    logger.info("Schema report written to %s", report_path)
