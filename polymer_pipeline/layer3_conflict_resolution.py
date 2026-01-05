import logging
from typing import List

import pandas as pd

from .utils import normalize_confidence, sanitize_smiles

logger = logging.getLogger("polymer_pipeline")


def _assign_confidence(row: pd.Series) -> str:
    has_repeat = bool(row.get("repeat_unit_smiles_full"))
    has_sidechain = bool(row.get("sidechain_smiles_list"))
    text_conf = normalize_confidence(row.get("confidence_text_norm"))
    source = row.get("source_priority", "text_name_only")

    if source == "OPSIN" and has_repeat:
        return "high"
    if has_repeat and has_sidechain and source == "image_pool":
        if text_conf == "high":
            return "high"
        return "medium"
    if has_repeat:
        return "medium"
    if has_sidechain:
        return "low"
    return "low"


def _assign_final_confidence(row: pd.Series) -> float:
    source = row.get("source_priority", "")
    base = 0.3
    if source == "OPSIN":
        base = 0.9
    elif source == "image_pool":
        base = 0.7
    elif source == "text_name_only":
        base = 0.5
    parse_conf = row.get("parse_confidence")
    if isinstance(parse_conf, (int, float)):
        base = max(base, float(parse_conf))
    elif isinstance(parse_conf, str):
        pc = normalize_confidence(parse_conf)
        if pc == "high":
            base = max(base, 0.9)
        elif pc == "medium":
            base = max(base, 0.7)
        elif pc == "low":
            base = max(base, 0.5)
    return round(base, 2)


def _decision_note(row: pd.Series) -> str:
    notes: List[str] = []
    if row.get("source_priority") == "image_pool":
        notes.append("image candidate selected")
    else:
        notes.append("text-derived only")
    if row.get("sidechain_smiles_list"):
        notes.append("sidechain inferred")
    if not row.get("repeat_unit_smiles_full"):
        notes.append("repeat missing")
    if row.get("backbone_smiles_text") and not row.get("repeat_unit_smiles_full"):
        notes.append("backbone only")
    if row.get("matching_decision"):
        notes.append(row["matching_decision"])
    return "; ".join(notes)


def run_conflict_resolution(df: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    Layer 3: finalize SMILES choices, assign confidence and decision notes.
    """
    resolved = df.copy()
    if not fast_mode:
        resolved["repeat_unit_smiles_full"] = resolved["repeat_unit_smiles_full"].apply(sanitize_smiles)
        resolved["backbone_smiles"] = resolved["backbone_smiles"].apply(sanitize_smiles)

    resolved["confidence"] = [_assign_confidence(row) for _, row in resolved.iterrows()]
    resolved["final_confidence"] = [_assign_final_confidence(row) for _, row in resolved.iterrows()]
    resolved["decision_note"] = [_decision_note(row) for _, row in resolved.iterrows()]

    logger.info("Layer 3 complete: assigned confidence to %d entries", len(resolved))
    return resolved
