import os
from pathlib import Path
from typing import Optional

import pandas as pd


def export_for_llm_review(
    df: pd.DataFrame,
    path: str,
    format: str = "jsonl",
    batch_size: int = 100,
    prompt_type: str = "Verify_SMILES",
) -> bool:
    """
    Export uncertain polymer rows for LLM-based review.
    """
    if df is None or df.empty or "needs_manual_review" not in df.columns:
        return False

    review_df = df[df["needs_manual_review"] == 1].copy()
    if review_df.empty:
        return False

    PROMPTS = {
        "Verify_SMILES": "Check whether predicted SMILES matches the polymer backbone and sidechain names. Correct it if necessary.",
        "Repair_Invalid_SMILES": "Attempt to repair invalid or incomplete SMILES while preserving polymer chemistry.",
        "Infer_Missing_Structure": "If SMILES is missing, infer a plausible polymer repeat unit based on backbone and sidechain information.",
    }
    prompt_text = PROMPTS.get(prompt_type, PROMPTS["Verify_SMILES"])
    review_df["llm_task_prompt"] = prompt_text

    # Map predicted_smiles column
    if "predicted_smiles" not in review_df.columns:
        # fall back to repeat_unit_smiles_full
        review_df["predicted_smiles"] = review_df.get("repeat_unit_smiles_full")

    # RDKit validation (optional)
    try:
        from rdkit import Chem

        def _check(sm):
            try:
                return bool(Chem.MolFromSmiles(str(sm)))
            except Exception:
                return False

    except Exception:
        def _check(sm):
            return bool(sm)

    review_df["smiles_valid"] = review_df["predicted_smiles"].apply(_check)
    review_df.loc[~review_df["smiles_valid"], "llm_task_prompt"] = PROMPTS["Repair_Invalid_SMILES"]

    total = len(review_df)
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(range(0, total, batch_size), start=1):
        batch = review_df.iloc[start : start + batch_size]
        fname = base.parent / f"{base.name}_batch_{i}.{format}"

        if format == "jsonl":
            batch.to_json(fname, orient="records", lines=True)
        elif format == "csv":
            batch.to_csv(fname, index=False)
        else:  # Markdown
            with open(fname, "w") as f:
                f.write("## LLM Task Options:\n")
                for key, val in PROMPTS.items():
                    f.write(f"- {key}: {val}\n")
                f.write("\n" + batch.to_markdown(index=False))
    return True
