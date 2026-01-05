import logging
import pandas as pd

from .utils import normalize_confidence, normalize_polymer_name, normalize_sidechain_name, sanitize_smiles

logger = logging.getLogger("polymer_pipeline")


def run_standardization(text_df: pd.DataFrame, image_df: pd.DataFrame):
    """
    Deterministic cleaning of textual and image-derived tables.
    Returns cleaned copies to avoid in-place mutation surprises.
    """
    text_clean = text_df.copy()
    image_clean = image_df.copy()

    # Normalize polymer names and related text fields.
    for col in ["polymer_name", "backbone_name_text", "sidechain_name_text", "paper_id", "sample_id"]:
        if col in text_clean.columns:
            text_clean[col] = text_clean[col].fillna("").apply(normalize_polymer_name)

    if "sidechain_name_text" in text_clean.columns:
        text_clean["sidechain_name_text_norm"] = text_clean["sidechain_name_text"].apply(normalize_sidechain_name)

    if "polymer_name" in text_clean.columns:
        text_clean["polymer_name_norm"] = text_clean["polymer_name"].apply(normalize_polymer_name)

    if "confidence_text" in text_clean.columns:
        text_clean["confidence_text_norm"] = text_clean["confidence_text"].apply(normalize_confidence)

    # Validate candidate SMILES in image pool.
    if "candidate_smiles" in image_clean.columns:
        image_clean["candidate_smiles_clean"] = image_clean["candidate_smiles"].apply(sanitize_smiles)
    else:
        image_clean["candidate_smiles_clean"] = None

    logger.info("Layer 0 complete: %d text rows, %d image rows", len(text_clean), len(image_clean))
    return text_clean, image_clean
