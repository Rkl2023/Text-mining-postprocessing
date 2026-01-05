import json
from pathlib import Path

import pandas as pd


def export_llm_ready(df: pd.DataFrame, output_path: Path):
    """Export a cleaned, LLM-friendly dataset for downstream review."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("export_llm_ready expects a DataFrame input")

    df_clean = df.replace(
        ["nan", "NaN", "None", "NULL", "null", "N/A", "auto-fixed", "???", "c?"],
        None,
    )
    records = df_clean.to_dict(orient="records")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
