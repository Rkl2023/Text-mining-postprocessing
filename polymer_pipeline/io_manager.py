import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from polymer_pipeline.utils import safe_json_dumps

logger = logging.getLogger("polymer_pipeline")

SUPPORTED_EXTENSIONS = [".parquet", ".csv", ".json", ".sqlite", ".db", ".pg", ".mongo"]
EXT_PRIORITY = {".parquet": 3, ".csv": 2, ".json": 1, ".sqlite": 1, ".db": 1, ".pg": 0, ".mongo": 0}

# Schema signatures (normalized column names)
POLYMER_SIGNATURES = [
    {"polymer_name", "backbone_name_text", "sidechain_name_text"},
    {"polymer_name", "backbone", "side_chain"},
    {"polymer", "backbone", "sidechain"},
]

IMAGE_SIGNATURES = [
    {"candidate_smiles", "fragment_id", "parse_status"},
    {"candidate_smiles", "image_id", "parse_status"},
    {"raw_smiles", "canonical_smiles", "status"},
    {"cleaned_smiles", "canonical_smiles", "segment_index"},
]


def _normalize_columns(columns: List[str]) -> List[str]:
    return [c.strip().lower().replace(" ", "_") for c in columns]


def _read_sample(path: Path, limit: int = 500) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path, nrows=limit, comment="#")
        if suffix == ".json":
            try:
                return pd.read_json(path, lines=True, nrows=limit)
            except Exception:
                return pd.read_json(path, nrows=limit)
        if suffix in {".sqlite", ".db"}:
            with sqlite3.connect(path) as conn:
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                if tables.empty:
                    return None
                table = tables["name"].iloc[0]
                return pd.read_sql(f"SELECT * FROM {table} LIMIT {limit}", conn)
    except Exception as exc:
        logger.debug("Failed reading sample %s: %s", path, exc)
        return None
    return None


def load_table(path: Path) -> pd.DataFrame:
    """
    Load a data file robustly and always return a pandas.DataFrame.
    Supports CSV/TSV, JSON, Parquet, SQLite.
    """
    def _force_dataframe(obj):
        import pandas as pd

        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.Series):
            df = obj.to_frame()
            if df.columns.size == 1 and not df.columns[0]:
                df.columns = ["value"]
            return df.reset_index(drop=True)
        if isinstance(obj, (list, dict)):
            return pd.DataFrame(obj).reset_index(drop=True)
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception:
                return pd.DataFrame()
        return obj.reset_index(drop=True)

    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            # Explicit delimiter/encoding to avoid column collapse or sniffing surprises.
            df = pd.read_csv(path, sep=",", engine="python", encoding="utf-8")
            # If the file collapsed to a single column, retry with common alternates.
            if df.shape[1] == 1:
                try:
                    df_alt = pd.read_csv(path, sep=";", engine="python", encoding="utf-8")
                    if df_alt.shape[1] > 1:
                        logger.info("Re-read %s using ';' delimiter (columns=%s)", path, list(df_alt.columns))
                        df = df_alt
                    else:
                        df_alt = pd.read_csv(path, sep="\t", engine="python", encoding="utf-8")
                        if df_alt.shape[1] > 1:
                            logger.info("Re-read %s using tab delimiter (columns=%s)", path, list(df_alt.columns))
                            df = df_alt
                except Exception:
                    pass
        elif suffix == ".json":
            try:
                df = pd.read_json(path, lines=True)
            except Exception:
                df = pd.read_json(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".sqlite", ".db"}:
            with sqlite3.connect(path) as conn:
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                if tables.empty:
                    raise ValueError(f"No tables found in {path}")
                table = tables["name"].iloc[0]
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
        else:
            raise ValueError(f"Unsupported file type: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load file {path}: {e}")

    df = _force_dataframe(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"load_table must return DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError(f"Empty table loaded from {path}")
    logger.info("Loaded %s with columns=%s shape=%s", path, list(df.columns), df.shape)
    return df


def _score_role(columns: List[str], role_signatures: List[set]) -> Tuple[float, set]:
    colset = set(_normalize_columns(columns))
    best_score = 0.0
    best_sig: set = set()
    for sig in role_signatures:
        matched = len(colset.intersection(sig))
        score = matched / len(sig)
        if score > best_score:
            best_score = score
            best_sig = sig
    return best_score, best_sig


def detect_input_files(data_dir: str = "data", autoload_log: str = "logs/autoload_report.json") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Automatically scans data_dir for polymer and image candidate files based on schema/header inspection.
    Returns (polymer_path, image_path).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        alt = Path("/data")
        if alt.exists():
            data_path = alt

    report_path = Path(autoload_log)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    candidates: List[Tuple[str, Path, float, int, int]] = []  # role, path, confidence, valid_rows, priority

    for path in data_path.glob("**/*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        sample = _read_sample(path)
        if sample is None or sample.empty:
            continue
        norm_cols = _normalize_columns(list(sample.columns))
        ext_prio = EXT_PRIORITY.get(suffix, 0)

        poly_score, poly_sig = _score_role(norm_cols, POLYMER_SIGNATURES)
        img_score, img_sig = _score_role(norm_cols, IMAGE_SIGNATURES)

        if poly_score > 0:
            confidence = min(1.0, poly_score + ext_prio * 0.05)
            valid_rows = int(sample.shape[0])
            candidates.append(("polymer", path, confidence, valid_rows, ext_prio))
            results[str(path)] = {
                "type": "polymer_data",
                "confidence": round(confidence, 3),
                "columns": norm_cols,
                "matched_signature": list(poly_sig),
            }
        if img_score > 0:
            confidence = min(1.0, img_score + ext_prio * 0.05)
            valid_rows = int(sample.shape[0])
            candidates.append(("image", path, confidence, valid_rows, ext_prio))
            results[str(path)] = {
                "type": "image_candidate_pool",
                "confidence": round(confidence, 3),
                "columns": norm_cols,
                "matched_signature": list(img_sig),
            }

    def _select(role: str) -> Optional[Path]:
        role_candidates = [c for c in candidates if c[0] == role]
        if not role_candidates:
            return None
        role_candidates.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
        return role_candidates[0][1]

    polymer_path = _select("polymer")
    image_path = _select("image")

    results["summary"] = {
        "selected_polymer_file": str(polymer_path) if polymer_path else None,
        "selected_image_file": str(image_path) if image_path else None,
    }
    safe_json_dumps(results, path=report_path)

    if polymer_path:
        logger.info("Detected polymer data: %s", polymer_path)
    else:
        logger.warning("No polymer data detected in %s", data_path)
    if image_path:
        logger.info("Detected image candidate pool: %s", image_path)
    else:
        logger.warning("No image candidate pool detected in %s", data_path)
    return polymer_path, image_path
