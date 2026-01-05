import logging

import pandas as pd

logger = logging.getLogger("polymer_pipeline")


def _needs_manual(row: pd.Series) -> int:
    # Trust OPSIN-derived parses entirely.
    if row.get("parse_source") == "OPSIN":
        return 0
    if not row.get("repeat_unit_smiles_full"):
        return 1
    if row.get("confidence") == "low":
        return 1
    if row.get("candidate_score") is None and row.get("source_priority") != "image_pool":
        # Relax for text-derived repeats with medium/high confidence.
        if row.get("confidence") in (None, "low"):
            return 1
        return 0
    return 0


def run_manual_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 4: flag entries requiring manual review.
    """
    updated = df.copy()
    updated["needs_manual_review"] = updated.apply(_needs_manual, axis=1)
    logger.info("Layer 4 complete: %d entries flagged for manual review", updated["needs_manual_review"].sum())
    return updated
