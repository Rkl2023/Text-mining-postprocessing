import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from polymer_pipeline.layer1_name_to_structure import BACKBONE_DICTIONARY, SIDECHAIN_DICTIONARY, name_to_smiles_via_opsin
try:
    from polymer_pipeline.utils.opsin_bridge import OPSIN_AVAILABLE
    import polymer_pipeline.utils.opsin_bridge as opsin_bridge
except Exception:
    OPSIN_AVAILABLE = False
    opsin_bridge = None
try:
    from polymer_pipeline.utils.opsin_bridge import combine_opsin_template_fragments
except Exception:
    combine_opsin_template_fragments = None
from polymer_pipeline.utils import (
    alkyl_smiles_from_length,
    sanitize_smiles,
    sidechain_text_to_length,
    safe_json_dumps,
    _fix_punctuation_errors,
    _normalize_sidechain_tokens,
    rdkit_tools,
)
from polymer_pipeline.utils.common import preprocess_opsin_name_sidechain_gentle, decode_sidechain_raw

logger = logging.getLogger("polymer_pipeline")
_SMILES_CANON_CACHE: Dict[str, Optional[str]] = {}
HEURISTIC_BEFORE_OPSIN = False
# CHANGELOG: OPSIN path now uses gentle preprocessing that preserves IUPAC syntax; no split parsing added.


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    if smiles in _SMILES_CANON_CACHE:
        return _SMILES_CANON_CACHE[smiles]
    if rdkit_tools.Chem:
        try:
            mol = rdkit_tools.Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                _SMILES_CANON_CACHE[smiles] = None
                return None
            canon = rdkit_tools.Chem.MolToSmiles(mol, canonical=True)
            _SMILES_CANON_CACHE[smiles] = canon
            return canon
        except Exception:
            _SMILES_CANON_CACHE[smiles] = None
            return None
    canon = sanitize_smiles(smiles)
    _SMILES_CANON_CACHE[smiles] = canon
    return canon


def _attempt_opsin(name: str, *, with_diag: bool = False):
    """
    Backward-compatible OPSIN attempt.
    - with_diag=False: returns canonical SMILES or None (legacy behavior)
    - with_diag=True: returns (smiles_or_none, diag_dict)
    """
    diag = {
        "input": name,
        "preprocessed": name,
        "cache_hit": None,
        "opsin_status": None,
        "opsin_message": None,
        "opsin_warnings": None,
        "exception": None,
    }
    if with_diag:
        try:
            from polymer_pipeline.utils.opsin_bridge import parse_chemical_name_with_diag
        except Exception as exc:
            diag["exception"] = f"import_error:{exc}"
            return None, diag
        try:
            smi_raw, bridge_diag = parse_chemical_name_with_diag(name)
            smi = _canonicalize_smiles(smi_raw) if smi_raw else None
            diag["cache_hit"] = bridge_diag.get("cached") if isinstance(bridge_diag, dict) else None
            diag["opsin_status"] = bridge_diag.get("status") if isinstance(bridge_diag, dict) else None
            diag["opsin_message"] = bridge_diag.get("message") if isinstance(bridge_diag, dict) else None
            diag["opsin_warnings"] = bridge_diag.get("warnings") if isinstance(bridge_diag, dict) else None
            diag["exception"] = bridge_diag.get("exception") if isinstance(bridge_diag, dict) else None
            return smi, diag
        except Exception as exc:
            diag["exception"] = f"{exc.__class__.__name__}:{exc}"
            return None, diag
    smi = name_to_smiles_via_opsin(name)
    if not smi:
        return None
    return _canonicalize_smiles(smi)


def _smiles_quality(smiles: Optional[str]) -> Dict[str, Optional[int]]:
    if not smiles:
        return {"has_dot": None, "dot_count": None, "len": None}
    dot_count = smiles.count(".")
    return {"has_dot": dot_count > 0, "dot_count": dot_count, "len": len(smiles)}


def _light_clean(text: str) -> str:
    text = _fix_punctuation_errors(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _preprocess_for_opsin_backbone_gentle(raw: str) -> str:
    """
    Gentle preprocessing that preserves IUPAC syntax: commas, brackets, parentheses, colons, primes.
    Only normalizes unicode dashes/primes and collapses whitespace.
    """
    if not isinstance(raw, str):
        return ""
    text = _fix_punctuation_errors(raw)
    text = text.replace("′", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    text = _opsin_ocr_fix_fused_ring_primes(text)
    return text


def _opsin_ocr_fix_fused_ring_primes(name: str) -> str:
    """
    Targeted OCR correction: in fused-ring brackets with locant pattern and a trailing letter0],
    replace that final 0 with a prime. Extremely conservative to avoid false positives.
    """
    if not isinstance(name, str) or "]" not in name or "[" not in name:
        return name

    def _fix_segment(seg: str) -> str:
        if ":" not in seg:
            return seg
        if not re.search(r"\d+,\d+-[A-Za-z]", seg):
            return seg
        return re.sub(r"(?<=-[A-Za-z])0(?=\])", "'", seg, count=1)

    parts = []
    last = 0
    for m in re.finditer(r"\[[^\]]+\]", name):
        parts.append(name[last:m.start()])
        segment = m.group(0)
        fixed = _fix_segment(segment)
        parts.append(fixed)
        last = m.end()
    parts.append(name[last:])
    return "".join(parts)


def _is_opsin_eligible_backbone(text: str) -> bool:
    """Conservative eligibility check on gentle text; preserves syntax."""
    if not isinstance(text, str):
        return False
    t = text.lower()
    strong_signals = [
        re.search(r"\d,\d", t),
        "[" in t or "]" in t or "(" in t or ")" in t or ":" in t,
        any(k in t for k in ["benzothiadiazole", "naphthalene", "indacenodithiophene", "dithieno", "thiadiazole", "imidazole", "pyridine", "diimide", "quinoxaline"]),
        re.search(r"(yl|diyl|one|amide|acid)s?$", t),
    ]
    if any(strong_signals):
        return True
    descriptive = any(k in t for k in ["based", "copolymer", "polymer", "donor", "acceptor", "unit", "backbone"])
    return not descriptive


def _is_opsin_eligible_sidechain(text: str) -> bool:
    """Conservative gate for sidechains to avoid OPSIN on obviously non-chemical strings."""
    if not isinstance(text, str):
        return False
    t = text.lower().strip()
    if any(w in t for w in ["conjugated", "sidechain", "branched", "containing", "terminated", "chain", "path", "group"]):
        return False
    if re.fullmatch(r"[a-z]{25,}", t):
        return False
    if re.search(r"doc\\d+", t):
        return False
    if len(t.split()) > 6:
        return False
    eligible_suffix = any(k in t for k in ["yl", "oxy", "thio", "phenyl", "trifluoro", "carbonyl", "ester", "glycol"])
    locant_pattern = bool(re.search(r"\\d-\\w", t))
    return (eligible_suffix or locant_pattern) and len(t) <= 40


def _expand_template_terms(cleaned: str) -> Tuple[str, List[str], bool]:
    """
    Expand shorthand tokens using BACKBONE_DICTIONARY keys to help OPSIN.
    No SMILES are generated; only textual substitutions are applied.
    """
    logs: List[str] = []
    expanded = cleaned
    expansion_map = {
        "dpp": "diketopyrrolopyrrole",
        "th": "thiophene",
        "tt": "thienothiophene",
        "bt": "benzothiadiazole",
        "bdt": "benzodithiophene",
        "fbt": "fluorobenzothiadiazole",
    }
    for key in sorted(BACKBONE_DICTIONARY.keys(), key=len, reverse=True):
        if key not in expansion_map:
            expansion_map[key] = key.replace("-", " ")
    expanded_any = False
    for token, repl in expansion_map.items():
        pattern = rf"\b{re.escape(token)}\b"
        if re.search(pattern, expanded):
            updated = re.sub(pattern, repl, expanded)
            if updated != expanded:
                logs.append(f"[TEMPLATE-EXPAND] replaced '{token}' -> '{repl}'")
                expanded = updated
                expanded_any = True
    return expanded, logs, expanded_any


LOCANT_RULES = {
    "thiophene": "2,5-",
    "bithiophene": "5,5'-",
    "diketopyrrolopyrrole": "3,6-",
    "benzodithiophene": "2,6-",
}


def _augment_locants(text: str) -> Tuple[str, bool, Optional[str], Optional[str]]:
    """
    Prefix known locants when missing. Returns (text, applied, base, locant_core).
    """
    applied = False
    matched_base = None
    locant_core = None
    if not isinstance(text, str) or re.search(r"\d", text):
        return text, applied, matched_base, locant_core
    for base, loc in LOCANT_RULES.items():
        pattern = rf"\b{re.escape(base)}\b"
        if re.search(pattern, text):
            locant_core = loc.rstrip("-")
            text = re.sub(pattern, f"{base}-{locant_core}", text)
            applied = True
            matched_base = base
            break
    return text, applied, matched_base, locant_core


def _append_diyl(text: str, base: Optional[str], locant_core: Optional[str]) -> Tuple[str, bool]:
    """
    Append '-diyl' after locants for polymerizable backbones.
    """
    if not base or not locant_core:
        return text, False
    if re.search(r"\bdiyl\b", text):
        return text, False
    target = f"{base}-{locant_core}"
    if target in text:
        return text.replace(target, f"{target}-diyl"), True
    return f"{text}-diyl", True


def _connector_to_diyl(text: str) -> str:
    text = re.sub(r"\b(and|with|vs)\b", "-diyl-", text, flags=re.IGNORECASE)
    text = text.replace("-alt-", "")
    text = re.sub(r"-diyl-+", "-diyl-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def _preprocess_for_opsin(raw_name: str, kind: str = "backbone") -> Tuple[str, List[str], Dict[str, bool]]:
    """
    Multi-stage normalization to make names more OPSIN-friendly.
    Returns the preprocessed string, log notes, and flag metadata.
    """
    logs: List[str] = []
    flags: Dict[str, bool] = {
        "punctuation_fixed": False,
        "locant_augmented": False,
        "diyl_appended": False,
        "template_expanded": False,
        "sidechain_token_expanded": False,
        "yl_appended": False,
        "split_tokens": False,
    }
    if not isinstance(raw_name, str):
        return "", logs, flags

    text = _fix_punctuation_errors(raw_name)
    flags["punctuation_fixed"] = text != raw_name

    # Stopword removal and base normalization
    text = clean_backbone_name(text)

    # Sidechain-specific adjustments: avoid diyl/locants, expand sidechain tokens, and append -yl
    if kind == "sidechain":
        # Expand common sidechain shorthand
        expanded = text
        for token, repl in SIDECHAIN_DICTIONARY.items():
            pattern = rf"\b{re.escape(token)}\b"
            updated = re.sub(pattern, repl, expanded)
            if updated != expanded:
                expanded = updated
                flags["sidechain_token_expanded"] = True
                continue
            # Fallback: replace embedded tokens (e.g., c8oxy)
            updated = re.sub(rf"{re.escape(token)}", repl, expanded)
            if updated != expanded:
                expanded = updated
                flags["sidechain_token_expanded"] = True
        # Split obvious concatenations to aid OPSIN tokenization
        if not re.search(r"\s", expanded) and len(expanded) > 12:
            expanded = re.sub(r"(sidechain)", " side chain ", expanded)
            expanded = re.sub(r"(conjugated)", " conjugated ", expanded)
            flags["split_tokens"] = True
        # Append -yl to alkyl/aryl/alkoxy looking tails when missing
        if not expanded.endswith("yl") and re.search(r"(alkyl|alkoxy|oxy|phenyl|thio|siloxane|yl)$", expanded) is None:
            if re.search(r"(octyl|decyl|hexyl|phenyl|oxy|alkoxy|thio|siloxane)$", expanded):
                expanded = expanded + "-yl"
                flags["yl_appended"] = True
        text = expanded
        expanded_any = False
        exp_logs = []
    else:
        # Locant augmentation then diyl append for backbone
        text, loc_applied, matched_base, locant_core = _augment_locants(text)
        if loc_applied:
            flags["locant_augmented"] = True
            text, diyl_added = _append_diyl(text, matched_base, locant_core)
            flags["diyl_appended"] = diyl_added
        expanded, exp_logs, expanded_any = _expand_template_terms(text)
        flags["template_expanded"] = expanded_any
        text = expanded

    expanded = text
    logs.append(f"[OPSIN-PREPROC] raw='{raw_name}' -> preprocessed='{expanded}'")
    if kind != "sidechain":
        logs.extend(exp_logs)
    return expanded, logs, flags


def preprocess_opsin_name(text: str, kind: str = "backbone") -> Tuple[str, Dict[str, bool], List[str]]:
    """
    Shared preprocessing wrapper for OPSIN inputs (backbone/sidechain).
    """
    if kind == "sidechain":
        normalized, flags = _normalize_sidechain_tokens(text)
        logs: List[str] = [f"[OPSIN-PREPROC] raw='{text}' -> preprocessed='{normalized}'"]
        return normalized, flags, logs
    preprocessed, logs, flags = _preprocess_for_opsin(text, kind=kind)
    return preprocessed, flags, logs


def _parse_backbone_with_stages(raw_name: str, polymer_name=None, paper_id=None, fast_mode: bool = False) -> Tuple[Optional[str], Optional[str], str, List[dict]]:
    traces: List[dict] = []
    if not raw_name:
        return None, None, "low", traces

    def log(stage, success, source=None, note="", smiles=None, cleaned=None, preprocessed=None, flags=None, diag=None, variant_kind=None, variant_step=None):
        qual = _smiles_quality(smiles)
        traces.append(
            {
                "paper_id": paper_id,
                "polymer_name": polymer_name,
                "raw_backbone": raw_name,
                "cleaned_backbone": cleaned,
                "preprocessed_for_opsin": preprocessed,
                "punctuation_fixed": bool(flags and flags.get("punctuation_fixed")),
                "locant_augmented": bool(flags and flags.get("locant_augmented")),
                "diyl_appended": bool(flags and flags.get("diyl_appended")),
                "stage": stage,
                "parse_source": source,
                "success": success,
                "note": note,
                "smiles": smiles,
                "opsin_status": (diag or {}).get("opsin_status") or (diag or {}).get("status") if diag else None,
                "opsin_message": (diag or {}).get("opsin_message") or (diag or {}).get("message") if diag else None,
                "opsin_warnings": (diag or {}).get("opsin_warnings") or (diag or {}).get("warnings") if diag else None,
                "smiles_has_dot": qual.get("has_dot"),
                "smiles_dot_count": qual.get("dot_count"),
                "variant_kind": variant_kind,
                "variant_step": variant_step,
                "opsin_eligible": bool(flags and flags.get("opsin_eligible")),
                "opsin_input_backbone": (diag or {}).get("opsin_input") if diag else None,
                "ocr_fix_applied": bool(diag and diag.get("ocr_fix_applied")),
                "ocr_fix_rule": (diag or {}).get("ocr_fix_rule") if diag else None,
                "ocr_fix_before": (diag or {}).get("ocr_fix_before") if diag else None,
                "ocr_fix_after": (diag or {}).get("ocr_fix_after") if diag else None,
            }
        )

    # Gentle OPSIN string (preserves brackets/commas)
    opsin_input = _preprocess_for_opsin_backbone_gentle(raw_name)
    ocr_applied = opsin_input != raw_name
    cleaned = clean_backbone_name(raw_name)  # retained for legacy fields/metrics only
    eligible = _is_opsin_eligible_backbone(opsin_input)
    eligibility_flags = {"opsin_eligible": eligible}
    diag_ocr = {
        "opsin_input": opsin_input,
        "ocr_fix_applied": ocr_applied,
        "ocr_fix_rule": "fused_ring_prime_0_to_quote" if ocr_applied else None,
        "ocr_fix_before": raw_name if ocr_applied else None,
        "ocr_fix_after": opsin_input if ocr_applied else None,
    }
    if not eligible:
        log("opsin_skip", False, "NON_OPSIN_ELIGIBLE", "ineligible", None, cleaned, opsin_input, eligibility_flags, diag_ocr, "skip", "eligibility")
        return None, "TEXT_FALLBACK", "low", traces

    # Stage 1: direct OPSIN on expanded name
    smi, diag = _attempt_opsin(opsin_input, with_diag=True)
    diag = diag or {}
    diag.update(diag_ocr)
    prep_flags = {**eligibility_flags}
    log("opsin_direct", bool(smi), "OPSIN" if smi else None, "", smi, cleaned, opsin_input, prep_flags, diag, "direct", None)

    if smi:
        return smi, "OPSIN", "high", traces

    # Stage 4: repair and retry
    if not fast_mode:
        repair_steps = [
            ("drop_descriptors", lambda t: re.sub(r"\b(polymer|copolymer|based|unit|moiety|donor|acceptor)\b", "", t, flags=re.IGNORECASE)),
            ("drop_substituted", lambda t: re.sub(r"\b(substituted|functionalized)\b", "", t, flags=re.IGNORECASE)),
            ("tidy_punct", lambda t: _light_clean(re.sub(r"-{2,}", "-", t))),
        ]
        repair_base = opsin_input
        for step_name, fn in repair_steps:
            repair_base = fn(repair_base)
            repair_base = repair_base.strip()
            if not repair_base:
                continue
            if not _is_opsin_eligible_backbone(repair_base):
                log("opsin_repair", False, None, f"{step_name}_ineligible", None, repair_base, opsin_input, prep_flags, None, "repair", step_name)
                continue
            smi, diag = _attempt_opsin(repair_base, with_diag=True)
            diag = diag or {}
            diag["opsin_input"] = repair_base
            log("opsin_repair", bool(smi), "OPSIN" if smi else None, step_name, smi, repair_base, repair_base, prep_flags, diag, "repair", step_name)
            if smi:
                return smi, "OPSIN", "high", traces

    # Stage 5: fallback
    log("fallback", False, "TEXT_FALLBACK", "no parse", None, cleaned, opsin_input, prep_flags)
    return None, "TEXT_FALLBACK", "low", traces


def _parse_sidechain_with_trace(name: str, polymer_name=None, paper_id=None) -> Tuple[Optional[str], str, str, bool, List[dict]]:
    traces: List[dict] = []
    # Gentle guard for obviously tokenized/invalid sidechains
    def _looks_tokenized(t: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{15,}", t)) or "sidechain" in t.lower()

    preprocessed, prep_flags, prep_logs = preprocess_opsin_name(name, kind="sidechain")
    cleaned = preprocessed
    if _looks_tokenized(cleaned):
        return None, cleaned, "NON_OPSIN_COMPATIBLE", False, traces
    template_logged = False

    def log(stage, success, source=None, smiles=None, note="", opsin_attempted=False):
        nonlocal template_logged
        if stage == "template_expand":
            template_logged = True
        traces.append(
            {
                "paper_id": paper_id,
                "polymer_name": polymer_name,
                "raw_sidechain": name,
                "cleaned_sidechain": cleaned,
                "preprocessed_for_opsin": preprocessed,
                "punctuation_fixed": bool(prep_flags.get("punctuation_fixed")),
                "locant_augmented": bool(prep_flags.get("locant_augmented")),
                "diyl_appended": bool(prep_flags.get("diyl_appended")),
                "template_expanded": bool(prep_flags.get("template_expanded")),
                "sidechain_token_expanded": bool(prep_flags.get("sidechain_token_expanded")),
                "yl_appended": bool(prep_flags.get("yl_appended")),
                "split_tokens": bool(prep_flags.get("split_tokens")),
                "opsin_attempted": opsin_attempted,
                "stage": stage,
                "parse_source": source,
                "success": success,
                "smiles": smiles,
                "note": note,
                "opsin_status": None,
                "opsin_message": None,
                "opsin_warnings": None,
                "decoded_type": None,
                "decoded_confidence": None,
                "decoded_needs_review": None,
                "decoded_components_json": None,
                "opsin_candidate": None,
            }
        )

    if prep_logs and not template_logged:
        for note in prep_logs:
            log("template_expand", True, None, None, note)

    if not cleaned:
        return None, cleaned, "", False, traces

    decoded = decode_sidechain_raw(name)
    if traces:
        traces[-1]["decoded_type"] = decoded.get("type")
        traces[-1]["decoded_confidence"] = decoded.get("confidence")
        traces[-1]["decoded_needs_review"] = decoded.get("needs_manual_review")
        traces[-1]["decoded_components_json"] = safe_json_dumps(decoded.get("components"))

    heuristic_smi = None
    length = sidechain_text_to_length(cleaned)
    if length:
        heuristic_smi = sanitize_smiles(alkyl_smiles_from_length(length))
        log("sidechain_length", bool(heuristic_smi), "ALKYL_LENGTH" if heuristic_smi else None, heuristic_smi)
        if HEURISTIC_BEFORE_OPSIN and heuristic_smi:
            traces[-1]["opsin_attempted"] = False
            return heuristic_smi, cleaned, "ALKYL_LENGTH", True, traces

    # Decoder-driven selection
    if heuristic_smi:
        return heuristic_smi, cleaned, "ALKYL_LENGTH", True, traces

    opsin_input = preprocess_opsin_name_sidechain_gentle(name)
    eligible = _is_opsin_eligible_sidechain(opsin_input)
    smi = None
    diag = {}

    # decoder smile use
    if decoded.get("smiles") and decoded.get("confidence") in {"high", "medium"}:
        return decoded.get("smiles"), opsin_input, "DECODER", True, traces

    # try OPSIN on candidate chunks if provided
    if decoded.get("opsin_candidates"):
        for cand in decoded["opsin_candidates"]:
            smi, diag = _attempt_opsin(cand, with_diag=True)
            log("sidechain_preprocessed_opsin", bool(smi), "SIDECHAIN_OPSIN" if smi else None, smi, opsin_attempted=True)
            if traces:
                traces[-1]["opsin_success"] = bool(smi)
                traces[-1]["opsin_candidate"] = cand
                traces[-1]["opsin_status"] = (diag or {}).get("status")
                traces[-1]["opsin_message"] = (diag or {}).get("message")
                traces[-1]["opsin_warnings"] = (diag or {}).get("warnings")
            if smi:
                return smi, cand, "SIDECHAIN_OPSIN", True, traces

    if eligible:
        smi, diag = _attempt_opsin(opsin_input, with_diag=True)
        log("sidechain_preprocessed_opsin", bool(smi), "SIDECHAIN_OPSIN" if smi else None, smi, opsin_attempted=True)
        if traces:
            traces[-1]["opsin_success"] = bool(smi)
            traces[-1]["opsin_status"] = (diag or {}).get("status")
            traces[-1]["opsin_message"] = (diag or {}).get("message")
            traces[-1]["opsin_warnings"] = (diag or {}).get("warnings")
    else:
        log("sidechain_non_opsin_compatible", False, "NON_OPSIN_COMPATIBLE", None, "ineligible")
    if smi:
        return smi, opsin_input, "SIDECHAIN_OPSIN", True, traces
    return None, opsin_input if opsin_input else cleaned, "DECODER" if decoded.get("type") != "unknown" else "", False, traces


def run_layer1_parse_text(text_df: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    Layer 1: Parse backbone/side-chain from flexible schema columns.
    """
    df = text_df.copy()
    parse_logs: List[dict] = []
    backbone_col = "backbone_name_text" if "backbone_name_text" in df.columns else "backbone"
    sidechain_col = "sidechain_name_text" if "sidechain_name_text" in df.columns else "side_chain"

    # Ensure backbone/sidechain columns exist to keep parsing alive even when raw data lacks them.
    if backbone_col not in df.columns:
        df[backbone_col] = df.get("backbone", "")
    if sidechain_col not in df.columns:
        df[sidechain_col] = df.get("side_chain", df.get("sidechain_name_text_norm", ""))

    backbone_smiles = []
    parse_sources: List[Optional[str]] = []
    parse_confidences: List[str] = []
    sidechain_smiles_list: List[List[str]] = []
    sidechain_sources: List[str] = []
    sidechain_cleaned: List[str] = []
    sidechain_success: List[bool] = []

    attempts = 0
    for _, row in df.iterrows():
        backbone_name = row.get(backbone_col, "")
        side_name = row.get(sidechain_col, "")
        if backbone_name:
            attempts += 1

        bb, source, pconf, traces = _parse_backbone_with_stages(
            backbone_name,
            polymer_name=row.get("polymer_name"),
            paper_id=row.get("paper_id"),
            fast_mode=fast_mode,
        )
        sc, sc_clean, sc_source, sc_success, sc_traces = _parse_sidechain_with_trace(
            side_name, polymer_name=row.get("polymer_name"), paper_id=row.get("paper_id")
        )
        backbone_smiles.append(bb)
        parse_sources.append(source)
        parse_confidences.append(pconf)
        sidechain_smiles_list.append([sc] if sc else [])
        sidechain_sources.append(sc_source)
        sidechain_cleaned.append(sc_clean)
        sidechain_success.append(sc_success)
        parse_logs.extend(traces)
        parse_logs.extend(sc_traces)

    logger.info(
        "Layer 1 input stats: backbone non-empty=%d sidechain non-empty=%d attempted_opsin=%d",
        sum(bool(str(b).strip()) for b in df[backbone_col]),
        sum(bool(str(s).strip()) for s in df[sidechain_col]),
        attempts,
    )
    # Quick sample of inputs for debugging
    logger.debug("Layer 1 sample backbone values: %s", list(df[backbone_col].head(5)))
    if "polymer_name" in df.columns:
        logger.debug("Layer 1 sample polymer_name values: %s", list(df["polymer_name"].head(5)))
    if backbone_smiles:
        logger.debug("Layer 1 sample parsed SMILES (first 5): %s", [b for b in backbone_smiles[:5] if b])
    # Show sample OPSIN inputs if available
    try:
        logger.debug("Layer 1 OPSIN input sample: %s", list(df[backbone_col].dropna().head(3)))
    except Exception:
        pass
    df["backbone_smiles_text"] = backbone_smiles
    df["parse_source"] = parse_sources
    df["parse_confidence"] = parse_confidences
    df["sidechain_smiles_list_text"] = sidechain_smiles_list
    df["sidechain_parse_source"] = sidechain_sources
    df["sidechain_cleaned_name"] = sidechain_cleaned
    df["sidechain_parse_success"] = sidechain_success
    # Store a single sidechain SMILES for convenience (first element if present)
    df["sidechain_smiles_text"] = [lst[0] if lst else None for lst in sidechain_smiles_list]
    logger.info(
        "Layer 1: parsed backbone for %d/%d entries; sidechains for %d",
        sum(1 for b in backbone_smiles if b),
        len(df),
        sum(1 for s in sidechain_smiles_list if s),
    )
    # Logging summary
    success_opsin = sum(1 for src in parse_sources if src and src.startswith("OPSIN"))
    success_template = sum(1 for src in parse_sources if src == "TEMPLATE")
    metrics = {
        "total": len(df),
        "opsin_success": success_opsin,
        "template_hits": success_template,
        "opsin_initialized": OPSIN_AVAILABLE,
    }
    if opsin_bridge:
        metrics.update(
            {
                "opsin_calls": getattr(opsin_bridge, "OPSIN_CALLS", 0),
                "opsin_cache_hits": getattr(opsin_bridge, "OPSIN_CACHE_HITS", 0),
                "opsin_cache_misses": getattr(opsin_bridge, "OPSIN_CACHE_MISSES", 0),
            }
        )
    Path("logs").mkdir(parents=True, exist_ok=True)
    safe_json_dumps(metrics, path="logs/opsin_metrics.json")
    trace_path = Path("logs/opsin_trace.csv")
    if parse_logs:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(parse_logs).to_csv(trace_path, index=False)
    return df
STOPWORDS = {
    "donor",
    "acceptor",
    "polymer",
    "film",
    "non-carbonyl",
    "noncarbonyl",
    "substituted",
    "backbone",
    "side-chain",
    "copolymer",
    "copolymers",
}


def clean_backbone_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    text = _fix_punctuation_errors(name).lower().strip()
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = re.sub(r"[;:,]+", " ", text)
    greek_map = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta"}
    for gk, gv in greek_map.items():
        text = text.replace(gk, gv)
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789"))
    text = re.sub(r"(?<=\d)(?=[a-zA-Z])", "-", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=\d)", "-", text)
    for word in STOPWORDS:
        text = re.sub(rf"\b{re.escape(word)}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Expand common bracketed abbreviations already handled by replacement above.
    for key in BACKBONE_DICTIONARY.keys():
        if f"({key})" in text:
            text = text.replace(f"({key})", f" {key} ")
    return text
