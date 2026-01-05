import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .io_manager import load_table
from .layers.layer0_normalize import validate_and_normalize
from .layers.layer1_parse_text import run_layer1_parse_text
from .layers.layer2_fusion import run_layer2_fusion
from .layer3_conflict_resolution import run_conflict_resolution
from .layer4_manual_review import run_manual_review
from .metrics import compute_metrics

logger = logging.getLogger("polymer_pipeline")


def run_pipeline(
    text_table_path: str,
    image_pool_path: str,
    output_path: str = "polymer_structures_groundtruth.csv",
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Execute the full pipeline from raw CSV inputs to final output.
    """
    logger.info("Loading data: %s and %s", text_table_path, image_pool_path)
    text_df = load_table(Path(text_table_path))
    image_df = load_table(Path(image_pool_path))
    logger.info("Polymer columns: %s", list(text_df.columns))
    logger.info("Image columns: %s", list(image_df.columns))
    logger.debug("Polymer head:\n%s", text_df.head(3))
    logger.debug("Image head:\n%s", image_df.head(3))
    # Quick sanity on expected keys
    for required in ["paper_id", "polymer_name"]:
        if required not in text_df.columns:
            logger.warning("Missing expected column in polymer table: %s", required)
    for required in ["paper_id", "candidate_smiles"]:
        if required not in image_df.columns:
            logger.warning("Missing expected column in image table: %s", required)
    text_df = pd.DataFrame(text_df)
    image_df = pd.DataFrame(image_df)
    for name, df in {"text_df": text_df, "image_df": image_df}.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be DataFrame, got {type(df)}")

    t0 = time.perf_counter()
    text_clean, image_clean = validate_and_normalize(text_df, image_df, schema_report_path=Path("logs/schema_report.json"))
    t1 = time.perf_counter()
    text_structured = run_layer1_parse_text(text_clean, fast_mode=fast_mode)
    t2 = time.perf_counter()
    matched = run_layer2_fusion(text_structured, image_clean, fast_mode=fast_mode)
    t3 = time.perf_counter()
    resolved = run_conflict_resolution(matched, fast_mode=fast_mode)
    t4 = time.perf_counter()
    reviewed = run_manual_review(resolved)
    t5 = time.perf_counter()

    metrics = compute_metrics(reviewed)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output_file, index=False)

    logger.info(
        "Pipeline complete. Layer timings (s): L0=%.2f L1=%.2f L2=%.2f L3=%.2f L4=%.2f total=%.2f",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
        t5 - t0,
    )
    try:
        import polymer_pipeline.utils.opsin_bridge as opsin_bridge

        logger.info(
            "OPSIN stats: calls=%s cache_hits=%s cache_misses=%s success=%s",
            getattr(opsin_bridge, "OPSIN_CALLS", 0),
            getattr(opsin_bridge, "OPSIN_CACHE_HITS", 0),
            getattr(opsin_bridge, "OPSIN_CACHE_MISSES", 0),
            getattr(opsin_bridge, "OPSIN_SUCCESS", 0),
        )
    except Exception:
        pass

    logger.info("Pipeline complete. Output written to %s", output_file.resolve())
    logger.info("Metrics summary: %s", metrics)
    return reviewed
