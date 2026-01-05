import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger("polymer_pipeline")


def compute_metrics(df: pd.DataFrame) -> dict:
    total = len(df) if len(df) else 1  # avoid division by zero

    coverage_sidechain = df["sidechain_smiles_list"].apply(lambda x: bool(x)).sum() / total
    coverage_backbone = df["backbone_smiles"].apply(lambda x: bool(x)).sum() / total
    coverage_repeat = df["repeat_unit_smiles_full"].apply(lambda x: bool(x)).sum() / total
    manual_rate = df["needs_manual_review"].sum() / total
    opsin_coverage = df["source_priority"].eq("OPSIN").sum() / total

    failure_reasons = Counter()
    for _, row in df.iterrows():
        if not row.get("repeat_unit_smiles_full"):
            failure_reasons["missing_repeat_unit"] += 1
        if row.get("needs_manual_review"):
            failure_reasons["manual_review"] += 1
        if not row.get("sidechain_smiles_list"):
            failure_reasons["missing_sidechain"] += 1

    metrics = {
        "coverage_sidechain": coverage_sidechain,
        "coverage_backbone": coverage_backbone,
        "coverage_repeat_unit": coverage_repeat,
        "manual_review_rate": manual_rate,
        "opsin_coverage": opsin_coverage,
        "top_failure_reasons": dict(failure_reasons.most_common()),
    }

    logger.info(
        "Metrics: sidechain=%.2f backbone=%.2f repeat=%.2f manual=%.2f",
        coverage_sidechain,
        coverage_backbone,
        coverage_repeat,
        manual_rate,
    )
    logger.info("Top failure reasons: %s", metrics["top_failure_reasons"])
    return metrics
