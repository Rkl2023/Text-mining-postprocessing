import logging
import os
import re
import sys
import time
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
from polymer_pipeline.layer2_candidate_matching import _extract_sidechains_from_candidate, _score_candidate
from polymer_pipeline.utils import (
    murcko_scaffold_smiles,
    sidechain_text_to_length,
    sanitize_smiles,
    rdkit_tools,
    count_aliphatic_chain_length,
)

logger = logging.getLogger("polymer_pipeline")
_FALLBACK_RECOVERIES = 0
_RD_PARSE_CACHE: Dict[str, Dict] = {}


def preclean_smiles(smiles: Optional[str]) -> Optional[str]:
    """
    Pure-Python cleanup: guard None/NaN, normalize OCR/dashes/R-groups, trim whitespace,
    prune to largest fragment, and drop overly long strings.
    """
    if smiles is None or not isinstance(smiles, str):
        return None
    text = smiles.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    # Basic replacements (Unicode dashes, odd equals)
    replacements = {
        "＝": "=",
        "–": "-",
        "—": "-",
        "‐": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # R-group normalization
    text = re.sub(r"\[R\d+[A-Za-z]*\]", "[*]", text)

    # OCR fixes / whitespace
    ocr_repairs = [
        (r"Cl(\d+)=", r"Cl="),
        (r"Br(\d+)=", r"Br="),
        (r"\s+", ""),
    ]
    for pattern, repl in ocr_repairs:
        text = re.sub(pattern, repl, text)

    # Fragment pruning: keep largest fragment by length as proxy
    fragments = [f.strip() for f in text.split(".") if f and f.strip()]
    if len(fragments) > 1:
        fragments = sorted(fragments, key=len, reverse=True)
        text = fragments[0]

    # Length guard
    if len(text) > 200:
        logger.warning("preclean_smiles: dropping overlong SMILES len=%d", len(text))
        return None
    return text


def validate_and_canonicalize(smiles: str) -> Tuple[Optional[str], Optional[object]]:
    """
    RDKit validation + canonicalization with caching.
    Returns (canonical_smiles, mol) or (None, None) on failure.
    """
    if smiles in _RD_PARSE_CACHE:
        cached = _RD_PARSE_CACHE[smiles]
        return cached.get("canonical"), cached.get("mol")

    rdkit_tools.ensure_rdkit_import()
    if rdkit_tools.Chem is None:
        return None, None
    Chem = rdkit_tools.Chem
    try:
        t0 = time.perf_counter()
        mol = Chem.MolFromSmiles(smiles)
        parse_dt = time.perf_counter() - t0
        if parse_dt > 2.0:
            logger.warning("MolFromSmiles slow (%.2fs) for SMILES len=%d fragments=%d", parse_dt, len(smiles), smiles.count("."))
        if mol is None:
            _RD_PARSE_CACHE[smiles] = {"canonical": None, "mol": None}
            return None, None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        _RD_PARSE_CACHE[smiles] = {"canonical": canonical, "mol": mol}
        return canonical, mol
    except Exception:
        _RD_PARSE_CACHE[smiles] = {"canonical": None, "mol": None}
        return None, None


def _find_best_image_candidate(row: pd.Series, image_df: pd.DataFrame):
    """
    Best-effort lookup in image pool when text/OPSIN parsing failed.
    """
    if image_df is None or image_df.empty:
        return None
    paper_id = str(row.get("paper_id") or "").strip().lower()
    polymer_name = row.get("polymer_name")
    matches = pd.DataFrame()

    if paper_id:
        matches = image_df[
            image_df.get("paper_id", "").astype(str).str.strip().str.lower() == paper_id
        ]
    if matches.empty and polymer_name:
        matches = image_df[
            image_df.get("polymer_name", "").astype(str) == str(polymer_name)
        ]
    if matches.empty and polymer_name:
        matches = image_df[
            image_df.get("polymer_name", "").astype(str).str.contains(str(polymer_name), na=False, regex=False)
        ]
    if matches.empty:
        return None

    def _candidate_smiles(row_local: pd.Series) -> Optional[str]:
        return (
            row_local.get("candidate_smiles_clean")
            or row_local.get("candidate_smiles")
            or row_local.get("canonical_smiles")
        )

    matches = matches.copy()
    matches["candidate_smiles_best"] = matches.apply(_candidate_smiles, axis=1)
    matches = matches[matches["candidate_smiles_best"].notna()]
    if matches.empty:
        return None
    # Reuse scoring if possible
    best_row = None
    best_score = float("-inf")
    expected_chain_len = sidechain_text_to_length(row.get("sidechain_name_text", ""))
    expect_eg = "eg" in str(row.get("sidechain_name_text", "")).lower() or "glycol" in str(
        row.get("sidechain_name_text", "")
    ).lower()
    expect_ionic = "ionic" in str(row.get("sidechain_name_text", "")).lower()
    for _, cand in matches.iterrows():
        score = _score_candidate(cand, expected_chain_len, expect_eg, expect_ionic)
        if score > best_score:
            best_score = score
            best_row = cand
    return best_row


def _best_key(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """Return (paper_id, polymer_name) best-effort for matching."""
    pid = row.get("paper_id") or row.get("paper_name")
    pname = row.get("polymer_name_norm") or row.get("polymer_name")
    if isinstance(pname, str):
        pname = pname.strip().lower()
    return pid, pname


def _prepare_candidate_features(image_df: pd.DataFrame):
    """
    Precompute canonical SMILES, Mol objects, scaffolds, and chain lengths once per unique candidate.
    Returns a tuple (processed_df, feature_cache).
    """
    feature_cache: Dict[str, Dict] = {}
    processed_rows: List[Dict] = []
    Chem = rdkit_tools.Chem
    if Chem is None:
        Chem = rdkit_tools.ensure_rdkit_import()

    def _first_nonempty(*vals):
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v
        return None

    def _is_complex(smiles: str) -> bool:
        if not isinstance(smiles, str):
            return False
        return len(smiles) > 200 or smiles.count(".") > 5

    def _compute(smiles: str) -> Dict:
        if not smiles or not isinstance(smiles, str):
            return {"canonical": None, "mol": None, "max_chain": None, "scaffold": None, "is_valid": False, "repaired": False}
        if smiles in _RD_PARSE_CACHE:
            return _RD_PARSE_CACHE[smiles]
        if smiles in feature_cache:
            return feature_cache[smiles]

        original = smiles
        cleaned = preclean_smiles(smiles)
        if not cleaned:
            feature_cache[smiles] = {
                "canonical": None,
                "mol": None,
                "max_chain": None,
                "scaffold": None,
                "is_valid": False,
                "repaired": False,
            }
            _RD_PARSE_CACHE[smiles] = feature_cache[smiles]
            return feature_cache[smiles]

        canonical = None
        mol = None
        max_chain = None
        scaffold = None
        is_valid = False

        if _is_complex(cleaned):
            logger.warning("Skipping heavy sanitization for complex SMILES (len=%d fragments=%d): %s", len(cleaned), cleaned.count("."), cleaned[:120])
            canonical = sanitize_smiles(cleaned) or cleaned
            is_valid = bool(canonical)
        else:
            canonical, mol = validate_and_canonicalize(cleaned)
            is_valid = bool(canonical)

        if canonical and mol is not None:
            scaffold = murcko_scaffold_smiles(canonical, mol=mol)
            max_chain = count_aliphatic_chain_length(canonical, mol=mol)
        elif canonical:
            max_chain = count_aliphatic_chain_length(canonical)

        feature_cache[smiles] = {
            "canonical": canonical or original,
            "mol": mol,
            "max_chain": max_chain,
            "scaffold": scaffold,
            "is_valid": is_valid,
            "repaired": False,
        }
        _RD_PARSE_CACHE[smiles] = feature_cache[smiles]
        if canonical and canonical != smiles:
            feature_cache[canonical] = feature_cache[smiles]
            _RD_PARSE_CACHE[canonical] = feature_cache[smiles]
        return feature_cache[smiles]

    for _, cand in image_df.iterrows():
        smiles = _first_nonempty(
            cand.get("candidate_smiles_clean"),
            cand.get("candidate_smiles"),
            cand.get("canonical_smiles"),
            cand.get("raw_smiles"),
            cand.get("cleaned_smiles"),
        )
        features = _compute(smiles)
        row_dict = cand.to_dict()
        if features.get("canonical") and features.get("canonical") != smiles:
            row_dict["candidate_smiles_clean"] = features.get("canonical")
        processed_rows.append({**row_dict, **{f"feature_{k}": v for k, v in features.items()}})

    return pd.DataFrame(processed_rows), feature_cache


def auto_correct_smiles(smiles: str) -> Optional[str]:
    """
    Simplified wrapper to preserve interface: preclean, then validate/canonicalize.
    """
    if smiles is None or not isinstance(smiles, str):
        return None
    if not smiles or str(smiles).lower() == "nan":
        return None
    auto_correct_smiles._last_rgroup_repaired = False  # type: ignore[attr-defined]
    auto_correct_smiles._last_fragment_repair = False  # type: ignore[attr-defined]
    auto_correct_smiles._last_fragment_repair_mode = None  # type: ignore[attr-defined]
    auto_correct_smiles._last_skipped_fragments = 0  # type: ignore[attr-defined]
    auto_correct_smiles._last_largest_heavy_atoms = 0  # type: ignore[attr-defined]

    # Emit one-time diagnostics to trace RDKit import context when repairs are skipped.
    if not getattr(auto_correct_smiles, "_diag_logged", False):
        logger.info(
            "[auto_correct_smiles] sys.executable=%s sys.path[0:3]=%s env(VIRTUAL_ENV)=%s env(CONDA_PREFIX)=%s env(PYTHONPATH)=%s",
            sys.executable,
            sys.path[:3],
            os.environ.get("VIRTUAL_ENV"),
            os.environ.get("CONDA_PREFIX"),
            os.environ.get("PYTHONPATH"),
        )
        auto_correct_smiles._diag_logged = True  # type: ignore[attr-defined]

    # Reuse cached results when available to avoid repeated parsing.
    cached = _RD_PARSE_CACHE.get(smiles)
    if cached:
        return cached.get("canonical")

    cleaned = preclean_smiles(smiles)
    if not cleaned:
        return None
    if isinstance(cleaned, str) and (len(cleaned) > 200 or cleaned.count(".") > 5):
        logger.warning("auto_correct_smiles: skipping heavy repair for complex SMILES len=%d fragments=%d", len(cleaned), cleaned.count("."))
        canonical_fast = sanitize_smiles(cleaned) or cleaned
        _RD_PARSE_CACHE[smiles] = {
            "canonical": canonical_fast,
            "mol": None,
            "max_chain": None,
            "scaffold": None,
            "is_valid": bool(canonical_fast),
            "repaired": False,
            "complex_skipped": True,
        }
        return canonical_fast

    canonical, mol = validate_and_canonicalize(cleaned)
    if canonical is None:
        return None
    _RD_PARSE_CACHE[smiles] = {
        "canonical": canonical,
        "mol": mol,
        "max_chain": None,
        "scaffold": None,
        "is_valid": True,
        "repaired": False,
    }
    return canonical


def run_layer2_fusion(text_df: pd.DataFrame, image_df: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    Layer 2: schema-aware fusion of text and image candidates.
    """
    matched: List[Dict] = []
    fallback_recoveries = 0
    repair_records: List[Dict] = []
    counts = {
        "attempted": 0,
        "repaired": 0,
        "repaired_candidates": 0,
        "repaired_used_in_output": 0,
        "repaired_rejected_by_reasonable_filter": 0,
        "polymer_name_matches_exact": 0,
        "polymer_name_matches_substring": 0,
        "paper_only_matches": 0,
        "name_only_matches_exact": 0,
        "name_only_matches_substring": 0,
        "no_name_matches": 0,
        "polymer_name_missing_text": 0,
        "polymer_name_missing_image": 0,
    }
    text_df = text_df.copy()
    image_df = image_df.copy()
    candidate_log_rows: List[Dict] = []
    counts["polymer_name_missing_text"] = int(text_df.get("polymer_name", pd.Series()).isna().sum())
    counts["polymer_name_missing_image"] = int(image_df.get("polymer_name", pd.Series()).isna().sum())
    for col in ["candidate_smiles_clean", "candidate_smiles", "canonical_smiles", "raw_smiles", "cleaned_smiles"]:
        if col in image_df.columns:
            image_df[col] = image_df[col].fillna("")
    logger.info(
        "[Fusion] Empty polymer_name rows: text=%s image=%s",
        counts["polymer_name_missing_text"],
        counts["polymer_name_missing_image"],
    )
    processed_image_df, feature_cache = _prepare_candidate_features(image_df)
    # Pre-normalize frequently used string columns once to avoid repeated .str calls.
    # Precompute normalized keys once to avoid repeating .str.lower/.str.strip inside the hot loop.
    img_paper_norm = processed_image_df.get("paper_id", pd.Series(index=processed_image_df.index, dtype=object))
    img_paper_norm = img_paper_norm.fillna("").astype(str).str.strip().str.lower()
    img_poly_norm = processed_image_df.get("polymer_name", pd.Series(index=processed_image_df.index, dtype=object))
    img_poly_norm = img_poly_norm.fillna("").astype(str).str.strip().str.lower()
    processed_image_df = processed_image_df.copy()
    processed_image_df["__paper_norm"] = img_paper_norm
    processed_image_df["__poly_norm"] = img_poly_norm

    def _normalize_val(val: Optional[str]) -> str:
        return str(val or "").strip().lower()

    # to_dict(records) is faster than iterrows and keeps deterministic row order.
    text_records = text_df.to_dict("records")

    for row in text_records:
        row_dict = dict(row)
        paper_key, poly_key = _best_key(row_dict)
        paper_key_norm = _normalize_val(paper_key)
        poly_key_norm = _normalize_val(poly_key)
        sidechain_text_val = row_dict.get("sidechain_name_text", "")
        sidechain_text_norm = str(sidechain_text_val or "")
        expected_chain_len = sidechain_text_to_length(sidechain_text_norm)
        expect_eg = "eg" in sidechain_text_norm.lower() or "glycol" in sidechain_text_norm.lower()
        expect_ionic = "ionic" in sidechain_text_norm.lower()

        # Candidate retrieval with flexible keys.
        candidates = pd.DataFrame()
        match_mode = None
        if "paper_id" in processed_image_df.columns and "polymer_name" in processed_image_df.columns and paper_key and poly_key:
            candidates = processed_image_df[
                (processed_image_df["__paper_norm"] == paper_key_norm)
                & (processed_image_df["__poly_norm"] == poly_key_norm)
            ]
            if not candidates.empty:
                match_mode = "exact_name"
                counts["polymer_name_matches_exact"] += len(candidates)
        if candidates.empty and "paper_id" in processed_image_df.columns and "polymer_name" in processed_image_df.columns and paper_key and poly_key:
            candidates = processed_image_df[
                (processed_image_df["__paper_norm"] == paper_key_norm)
                & processed_image_df["__poly_norm"].str.contains(poly_key_norm, na=False, regex=False)
            ]
            if not candidates.empty:
                match_mode = "substring_name"
                counts["polymer_name_matches_substring"] += len(candidates)
        if candidates.empty and "paper_id" in processed_image_df.columns and paper_key:
            candidates = processed_image_df[(processed_image_df["__paper_norm"] == paper_key_norm)]
            if not candidates.empty and match_mode is None:
                match_mode = "paper_only"
                counts["paper_only_matches"] += len(candidates)
        if candidates.empty and "polymer_name" in processed_image_df.columns and poly_key:
            candidates = processed_image_df[processed_image_df["__poly_norm"] == poly_key_norm]
            if not candidates.empty and match_mode is None:
                match_mode = "name_only_exact"
                counts["name_only_matches_exact"] += len(candidates)
        if candidates.empty and "polymer_name" in processed_image_df.columns and poly_key:
            candidates = processed_image_df[processed_image_df["__poly_norm"].str.contains(poly_key_norm, na=False, regex=False)]
            if not candidates.empty and match_mode is None:
                match_mode = "name_only_substring"
                counts["name_only_matches_substring"] += len(candidates)
        if candidates.empty and match_mode is None:
            match_mode = "no_match"
            counts["no_name_matches"] += 1

        best_smiles = None
        best_score = float("-inf")
        decision_bits: List[str] = []
        sidechains = list(row_dict.get("sidechain_smiles_list_text", []))
        best_log_index: Optional[int] = None

        # Validate candidates using RDKit when available.
        def canonical_smiles(sm):
            if not sm:
                return None
            counts["attempted"] += 1
            features = feature_cache.get(sm) or _RD_PARSE_CACHE.get(sm) or {}
            candidate_canon = features.get("canonical")
            if candidate_canon:
                return candidate_canon
            cleaned = preclean_smiles(sm)
            if not cleaned:
                return None
            canon, mol = validate_and_canonicalize(cleaned)
            if canon:
                counts["repaired"] += 1
                repair_records.append(
                    {
                        "paper_id": paper_key,
                        "polymer_name": poly_key,
                        "original_smiles": sm,
                        "repaired_smiles": canon,
                        "status": "auto_fixed",
                    }
                )
                feature_cache[sm] = {
                    "canonical": canon,
                    "mol": mol,
                    "max_chain": count_aliphatic_chain_length(canon, mol=mol) if mol else count_aliphatic_chain_length(canon),
                    "scaffold": murcko_scaffold_smiles(canon, mol=mol),
                    "is_valid": True,
                    "repaired": False,
                }
                feature_cache[canon] = feature_cache[sm]
                _RD_PARSE_CACHE[sm] = feature_cache[sm]
                _RD_PARSE_CACHE[canon] = feature_cache[sm]
                return canon
            return None

        def _is_reasonable_smiles(sm):
            if not sm or not isinstance(sm, str):
                return False
            s = sm.strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return False
            if rdkit_tools.Chem is None:
                return True
            features = feature_cache.get(sm, {})
            mol = features.get("mol")
            try:
                if mol is None:
                    mol = rdkit_tools.Chem.MolFromSmiles(s)
                if mol is None:
                    return False
                if mol.GetNumHeavyAtoms() < 4:
                    return False
                return True
            except Exception:
                return False

        for cand in candidates.to_dict("records"):
            smiles = cand.get("candidate_smiles_clean") or cand.get("candidate_smiles") or cand.get("canonical_smiles")
            if not smiles:
                candidate_log_rows.append(
                    {
                        "paper_id": paper_key,
                        "polymer_name_text": poly_key,
                        "candidate_smiles": None,
                        "raw_candidate_smiles": None,
                        "candidate_score": None,
                        "parse_status": cand.get("parse_status"),
                        "feature_is_valid": None,
                        "feature_repaired": None,
                        "match_mode": match_mode,
                        "was_selected": False,
                        "reason_filtered": "missing_smiles",
                    }
                )
                continue
            features = feature_cache.get(smiles) or {}
            canonical = features.get("canonical") or canonical_smiles(smiles)
            if not canonical:
                candidate_log_rows.append(
                    {
                        "paper_id": paper_key,
                        "polymer_name_text": poly_key,
                        "candidate_smiles": None,
                        "raw_candidate_smiles": smiles,
                        "candidate_score": None,
                        "parse_status": cand.get("parse_status"),
                        "feature_is_valid": features.get("is_valid"),
                        "feature_repaired": features.get("repaired"),
                        "match_mode": match_mode,
                        "was_selected": False,
                        "reason_filtered": "invalid_smiles",
                    }
                )
                continue
            # Enrich candidate fields for scoring if RDKit parsed
            mol = feature_cache.get(smiles, {}).get("mol")
            if mol is None and canonical:
                _, mol = validate_and_canonicalize(canonical)
            if mol is not None:
                if not cand.get("heavy_atom_count"):
                    cand["heavy_atom_count"] = mol.GetNumHeavyAtoms()
                if not cand.get("parse_status"):
                    cand["parse_status"] = "ok"
                if not cand.get("max_aliphatic_chain_len"):
                    try:
                        cand["max_aliphatic_chain_len"] = count_aliphatic_chain_length(canonical, mol=mol)
                    except Exception:
                        pass
                cand["canonical_smiles"] = canonical
            score = _score_candidate(cand, expected_chain_len, expect_eg, expect_ionic)
            if score > best_score:
                best_score = score
                best_smiles = canonical
                best_log_index = len(candidate_log_rows)
            candidate_log_rows.append(
                {
                    "paper_id": paper_key,
                    "polymer_name_text": poly_key,
                    "candidate_smiles": canonical or smiles,
                    "raw_candidate_smiles": smiles,
                    "candidate_score": score,
                    "parse_status": cand.get("parse_status"),
                    "feature_is_valid": features.get("is_valid"),
                    "feature_repaired": features.get("repaired"),
                    "match_mode": match_mode,
                    "was_selected": False,
                    "reason_filtered": "pending",
                }
            )

        repaired_flag = best_smiles and feature_cache.get(best_smiles, {}).get("repaired")
        if best_smiles and _is_reasonable_smiles(best_smiles):
            decision_bits.append("image_pool_match")
            sidechains.extend(_extract_sidechains_from_candidate(best_smiles))
            if repaired_flag:
                counts["repaired_used_in_output"] += 1
        else:
            decision_bits.append("no_image_match")
            if best_smiles and repaired_flag:
                counts["repaired_rejected_by_reasonable_filter"] += 1
            if best_log_index is not None and best_log_index < len(candidate_log_rows):
                candidate_log_rows[best_log_index]["reason_filtered"] = "not_reasonable"
                candidate_log_rows[best_log_index]["was_selected"] = False
            best_smiles = None

        if best_smiles is None:
            fallback_candidate = _find_best_image_candidate(row, processed_image_df)
            if fallback_candidate is not None:
                cand_smiles = (
                    fallback_candidate.get("candidate_smiles_clean")
                    or fallback_candidate.get("candidate_smiles")
                    or fallback_candidate.get("canonical_smiles")
                )
                cand_smiles = cand_smiles or ""
                cand_smiles = sanitize_smiles(cand_smiles) or cand_smiles
                if cand_smiles and _is_reasonable_smiles(cand_smiles):
                    best_smiles = cand_smiles
                    decision_bits.append("image_pool_match")
                    fallback_recoveries += 1

        if best_log_index is not None and best_log_index < len(candidate_log_rows) and best_smiles:
            candidate_log_rows[best_log_index]["was_selected"] = True
            candidate_log_rows[best_log_index]["reason_filtered"] = "selected"
        # Any remaining pending rows are lower_score (or unselected)
        for row_idx, log_row in enumerate(candidate_log_rows):
            if log_row.get("reason_filtered") == "pending" and log_row.get("was_selected") is False:
                candidate_log_rows[row_idx]["reason_filtered"] = "lower_score"

        # Priority: OPSIN > image_pool > unknown (text-derived or missing)
        repeat_unit_smiles_full = None
        backbone_smiles = None
        source_priority = "unknown"

        parse_source_val = row_dict.get("parse_source")
        if parse_source_val == "OPSIN" and row_dict.get("backbone_smiles_text"):
            repeat_unit_smiles_full = row_dict.get("backbone_smiles_text")
            backbone_smiles = row_dict.get("backbone_smiles_text")
            source_priority = "OPSIN"
        elif best_smiles:
            repeat_unit_smiles_full = best_smiles
            cached_scaffold = feature_cache.get(best_smiles, {}).get("scaffold") if feature_cache else None
            backbone_smiles = cached_scaffold or murcko_scaffold_smiles(best_smiles) if best_smiles else None
            source_priority = "image_pool"
        elif row_dict.get("backbone_smiles_text"):
            repeat_unit_smiles_full = row_dict.get("backbone_smiles_text")
            backbone_smiles = row_dict.get("backbone_smiles_text")
            source_priority = "unknown"

        matched.append(
            {
                **row_dict,
                "repeat_unit_smiles_full": repeat_unit_smiles_full,
                "backbone_smiles": backbone_smiles,
                "sidechain_smiles_list": list(dict.fromkeys([s for s in sidechains if s])),
                "source_priority": source_priority,
                "image_candidate_smiles": best_smiles,
                "matching_decision": ";".join(decision_bits),
                "candidate_score": best_score if best_smiles else None,
                "parse_source": parse_source_val,
                "match_mode": match_mode,
            }
        )

    logger.info("Layer 2: fused %d entries (matched repeat units for %d)", len(matched), sum(1 for r in matched if r.get("repeat_unit_smiles_full")))
    if fallback_recoveries:
        logger.info("Layer 2: image-pool fallback recoveries: %d", fallback_recoveries)
    if repair_records:
        log_path = Path("logs/repair_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(repair_records).to_csv(log_path, index=False)
    if counts["attempted"]:
        pct = counts["repaired"] / counts["attempted"] * 100
        logger.info("Auto-corrected SMILES: %d/%d (%.1f%%) repaired", counts["repaired"], counts["attempted"], pct)
    logger.info(
        "Repaired candidates: %d used_in_output=%d rejected_by_filter=%d",
        counts["repaired_candidates"],
        counts["repaired_used_in_output"],
        counts["repaired_rejected_by_reasonable_filter"],
    )
    fusion_match_metrics = {
        "polymer_name_matches_exact": counts["polymer_name_matches_exact"],
        "polymer_name_matches_substring": counts["polymer_name_matches_substring"],
        "paper_only_matches": counts["paper_only_matches"],
        "name_only_matches_exact": counts["name_only_matches_exact"],
        "name_only_matches_substring": counts["name_only_matches_substring"],
        "polymer_name_missing_text": counts["polymer_name_missing_text"],
        "polymer_name_missing_image": counts["polymer_name_missing_image"],
        "no_name_matches": counts["no_name_matches"],
        "unknown_remaining_after_name_match": sum(1 for r in matched if r.get("source_priority") == "unknown"),
    }
    log_path = Path("logs/fusion_match_metrics.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(fusion_match_metrics, indent=2))
    if candidate_log_rows:
        cand_log_path = Path("logs/candidate_scores.csv")
        cand_log_path.parent.mkdir(parents=True, exist_ok=True)
        cand_df = pd.DataFrame(candidate_log_rows)
        cand_df.to_csv(cand_log_path, mode="a", header=not cand_log_path.exists(), index=False)
    return pd.DataFrame(matched)
