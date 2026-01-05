import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .utils import (
    extract_longest_aliphatic_chain,
    count_aliphatic_chain_length,
    murcko_scaffold_smiles,
    parse_boolean,
    sanitize_smiles,
    sidechain_text_to_length,
)

logger = logging.getLogger("polymer_pipeline")
_CHAIN_CACHE: Dict[str, int] = {}


def _cached_chain_len(smiles: str) -> int:
    if not smiles:
        return 0
    if smiles in _CHAIN_CACHE:
        return _CHAIN_CACHE[smiles]
    length = count_aliphatic_chain_length(smiles)
    _CHAIN_CACHE[smiles] = length
    return length


def _score_candidate(
    row: pd.Series,
    expected_chain_len: Optional[int],
    expect_eg: bool,
    expect_ionic: bool,
) -> float:
    score = 0.0

    parse_status = str(row.get("parse_status", "")).lower()
    canonical = row.get("canonical_smiles")
    parsed_ok = parse_status in {"ok", "success", "parsed"} or (isinstance(canonical, str) and canonical.strip() != "")
    has_metal = parse_boolean(row.get("has_metal"))
    has_counterion = parse_boolean(row.get("has_counterion"))
    heavy_atoms = row.get("heavy_atom_count")
    eg_count = row.get("eg_motif_count")
    max_chain = row.get("max_aliphatic_chain_len") or _cached_chain_len(row.get("candidate_smiles_clean", ""))

    if parsed_ok:
        score += 2
    if has_metal:
        score -= 4
    if has_counterion:
        score -= 3
    if isinstance(heavy_atoms, (int, float)):
        score += min(heavy_atoms / 50.0, 2.0)

    if expected_chain_len and max_chain:
        diff = abs(max_chain - expected_chain_len)
        score += max(0, 2 - diff * 0.5)
    elif expected_chain_len and not max_chain:
        score -= 0.5

    if expect_eg and eg_count:
        score += min(eg_count, 3) * 0.5
    if expect_ionic and parse_status in {"charged", "ionic"}:
        score += 1

    return score


def _extract_sidechains_from_candidate(smiles: str) -> List[str]:
    """
    Very coarse heuristic: if a long aliphatic chain is present, keep it as side-chain.
    """
    chains: List[str] = []
    length = _cached_chain_len(smiles)
    if length and length >= 6:
        chain_smiles = "C" * length
        chain_clean = sanitize_smiles(chain_smiles)
        if chain_clean:
            chains.append(chain_clean)
    return chains


def run_candidate_matching(
    text_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Layer 2: Match text records with image-derived candidates.
    """
    matched_rows: List[Dict] = []

    for _, row in text_df.iterrows():
        paper_id = row.get("paper_id", row.get("sample_id", ""))
        polymer_name = (row.get("polymer_name_norm") or row.get("polymer_name") or "").strip().lower()
        expected_chain_len = sidechain_text_to_length(row.get("sidechain_name_text_norm", ""))
        expect_eg = "eg" in str(row.get("sidechain_name_text_norm", "")).lower() or "glycol" in str(
            row.get("sidechain_name_text_norm", "")
        ).lower()
        expect_ionic = "ionic" in str(row.get("sidechain_name_text_norm", "")).lower()

        candidates = image_df[image_df["paper_id"] == paper_id] if "paper_id" in image_df.columns else pd.DataFrame()
        best_candidate_smiles = None
        best_score = float("-inf")
        decision_bits: List[str] = []
        sidechains = list(row.get("sidechain_smiles_list_text", []))

        for _, cand in candidates.iterrows():
            smiles = cand.get("candidate_smiles_clean")
            if not smiles:
                continue
            score = _score_candidate(cand, expected_chain_len, expect_eg, expect_ionic)
            if score > best_score:
                best_score = score
                best_candidate_smiles = smiles

        if best_candidate_smiles:
            decision_bits.append("image_pool_match")
            sidechains.extend(_extract_sidechains_from_candidate(best_candidate_smiles))
        else:
            decision_bits.append("no_image_match")

        repeat_unit_smiles_full = best_candidate_smiles or row.get("backbone_smiles_text")
        backbone_smiles = murcko_scaffold_smiles(best_candidate_smiles) if best_candidate_smiles else row.get("backbone_smiles_text")

        matched_rows.append(
            {
                **row.to_dict(),
                "repeat_unit_smiles_full": repeat_unit_smiles_full,
                "backbone_smiles": backbone_smiles,
                "sidechain_smiles_list": list(dict.fromkeys([s for s in sidechains if s])),
                "source_priority": "image_pool" if best_candidate_smiles else "text_name_only",
                "matching_decision": ";".join(decision_bits),
                "candidate_score": best_score if best_candidate_smiles else None,
            }
        )

    logger.info("Layer 2 complete: matched %d/%d polymers to image pool", sum(1 for r in matched_rows if r.get("repeat_unit_smiles_full")), len(matched_rows))
    return pd.DataFrame(matched_rows)
