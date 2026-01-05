import json
import numpy as np
from datetime import datetime, date
import re
import unicodedata

try:
    from rdkit import Chem
except Exception:
    Chem = None

SIDECHAIN_TOKEN_MAP = {
    "c1": "methyl",
    "c2": "ethyl",
    "c3": "propyl",
    "c4": "butyl",
    "c5": "pentyl",
    "c6": "hexyl",
    "c7": "heptyl",
    "c8": "octyl",
    "c9": "nonyl",
    "c10": "decyl",
    "doc10": "2-decyldodecyl",
    "tetrac10": "tetradecyl",
    "oxy": "oxy",
    "alkoxy": "alkoxy",
    "ethoxy": "ethoxy",
    "methoxy": "methoxy",
    "ph": "phenyl",
    "phenyl": "phenyl",
    "benzyl": "benzyl",
    "eg": "ethoxyethyl",
    "peg": "poly(ethylene glycol)",
    "thio": "thio",
    "alkylthio": "alkylthio",
    "siloxane": "siloxane",
    "t-boc": "tert-butoxycarbonyl",
}

def safe_json_dumps(obj, **kwargs):
    """
    Safely serialize Python or pandas/numpy objects to JSON-compatible string.
    Handles np.int64, np.float64, NaN, datetime, sets, and ndarrays.
    """

    class EnhancedEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            if isinstance(o, set):
                return list(o)
            return super().default(o)

    try:
        return json.dumps(obj, cls=EnhancedEncoder, **kwargs)
    except Exception:
        return json.dumps(str(obj))


def normalize_confidence(value):
    """
    Normalize confidence field to canonical string levels:
    'high', 'medium', 'low', or 'unknown'.
    Supports numeric or string inputs.
    """
    if value is None:
        return "unknown"

    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.4:
            return "medium"
        if value >= 0:
            return "low"
        return "unknown"

    val = str(value).strip().lower()
    if val in {"h", "hi", "high", "1.0", "very high"}:
        return "high"
    if val in {"m", "med", "medium"}:
        return "medium"
    if val in {"l", "lo", "low", "weak"}:
        return "low"
    return "unknown"


def normalize_confidence_series(series):
    return series.apply(normalize_confidence)


def _fix_punctuation_errors(text: str) -> str:
    """
    Normalize common OCR/encoding punctuation issues while preserving valid '-' and apostrophes.
    """
    if not isinstance(text, str):
        return ""
    original = text
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove odd dash/comma combos and duplicated punctuation
    text = re.sub(r"-,|,-", "-", text)
    text = re.sub(r"[\\.]{2,}", " ", text)
    text = re.sub(r",,", " ", text)
    text = re.sub(r"[;]+", " ", text)
    # Trim trailing punctuation (commas/periods/semicolons)
    text = re.sub(r"[\\.,;:]+$", "", text.strip())
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _normalize_sidechain_tokens(text: str):
    """
    Convert common sidechain shorthand into more OPSIN-friendly fragments.
    """
    flags = {"tokens_normalized": False, "yl_appended": False, "punctuation_fixed": False, "opsin_ready": False}
    if not isinstance(text, str):
        return "", flags
    original = text
    t = _fix_punctuation_errors(text)
    flags["punctuation_fixed"] = t != original
    t = t.lower().replace("_", "-")
    t = re.sub(r"--+", "-", t)
    for k, v in SIDECHAIN_TOKEN_MAP.items():
        if k in t:
            t_new = t.replace(k, v)
            if t_new != t:
                t = t_new
                flags["tokens_normalized"] = True
    t = re.sub(r"([a-z])([0-9])", r"\1-\2", t)
    t = re.sub(r"([0-9])([a-z])", r"\1-\2", t)
    for bad in ["sidechain", "unit", "donor", "acceptor", "group", "onbdt", "onntz", "onisoindigo"]:
        t = t.replace(bad, " ")
    t = re.sub(r"\\s+", " ", t).strip(" -")
    t = t.replace(" and ", "-yl-").replace(" with ", "-yl-").replace(" vs ", "-yl-")
    if not re.search(r"(yl|oxy|phenyl|thio)$", t):
        t = t + "-yl"
        flags["yl_appended"] = True
    flags["opsin_ready"] = True
    return t.strip(), flags


def preprocess_opsin_name_sidechain_gentle(raw: str) -> str:
    """
    OPSIN-only gentle sidechain preprocessing:
    - preserve hyphens/digits/brackets/commas/parentheses
    - unicode dash/primes normalization, underscores->hyphen, collapse spaces, trim
    - no forced -yl, no connector rewrites, no aggressive stopword removal
    """
    if not isinstance(raw, str):
        return ""
    t = _fix_punctuation_errors(raw)
    t = t.replace("′", "'").replace("’", "'")
    t = t.replace("_", "-")
    t = re.sub(r"\s+", " ", t).strip(" .")
    return t


def sidechain_preclean(raw: str) -> str:
    """Light cleaning for raw sidechain decoder; preserves punctuation and parentheses."""
    return preprocess_opsin_name_sidechain_gentle(raw)


def extract_parenthetical_chunks(raw: str):
    """Return list of inner strings from parenthetical chunks."""
    if not isinstance(raw, str):
        return []
    return [m.group(1) for m in re.finditer(r"\(([^)]{3,})\)", raw)]


def gen_linear_alkyl(n: int) -> str:
    """Generate attachment-marked linear alkyl."""
    try:
        n = int(n)
    except Exception:
        return ""
    if n < 1:
        return ""
    return "[*]" + "C" * n


_BRANCHED_ALKYL_MAP = {
    "2-ethylhexyl": "CCCCC(C)CC[*]" .replace("[*]", ""),  # placeholder pattern; not used directly
}
# explicit high-confidence branch mappings to SMILES with attachment [*]
_ALKYL_SMILES_MAP = {
    "2-ethylhexyl": "[*]CCCCC(C)CC",
    "2-octyldodecyl": "[*]CCCCCCCCCCCC(C)CCCCCCCC",
    "2-hexyldecyl": "[*]CCCCCCCCCC(C)CCCCCC",
    "2-butyloctyl": "[*]CCCCCCCC(C)CCCCC",
    "hexyl": "[*]CCCCCC",
    "octyl": "[*]CCCCCCCC",
    "dodecyl": "[*]CCCCCCCCCCCC",
    "2-ethylhexan-1-yl": "[*]CCCCC(C)CC",
}


def decode_sidechain_raw(raw: str):
    """
    Conservative raw sidechain decoder.
    Returns dict with fields: type, components, smiles, confidence, needs_manual_review, notes, opsin_candidates.
    """
    result = {
        "type": "unknown",
        "components": [],
        "smiles": None,
        "confidence": "low",
        "needs_manual_review": True,
        "notes": "",
        "opsin_candidates": [],
    }
    if not isinstance(raw, str):
        result["notes"] = "non_string"
        return result
    text = sidechain_preclean(raw)
    low = text.lower()

    # formula parser
    m = re.search(r"n?-?c(\d{1,2})h\d+", low)
    if m:
        ccount = int(m.group(1))
        smi = gen_linear_alkyl(ccount)
        if smi:
            result.update(
                {
                    "type": "formula_linear",
                    "components": [{"name": text, "count": 1}],
                    "smiles": smi,
                    "confidence": "high",
                    "needs_manual_review": False,
                    "notes": "formula",
                }
            )
            return result

    # exact mapping
    for key, smi in _ALKYL_SMILES_MAP.items():
        if low.strip() == key:
            result.update(
                {
                    "type": "mapped_linear_or_branched",
                    "components": [{"name": text, "count": 1}],
                    "smiles": smi,
                    "confidence": "high",
                    "needs_manual_review": False,
                    "notes": "exact_map",
                }
            )
            return result

    # multiplicity patterns
    if re.search(r"\btwo\b", low) and re.search(r"hexyl|octyl|ethyl", low):
        result.update(
            {
                "type": "multiplicity",
                "components": [{"name": text, "count": 2}],
                "confidence": "medium",
                "needs_manual_review": True,
                "notes": "multiplicity_detected",
            }
        )
        return result
    if low.startswith("di") and low[2:].endswith("yl"):
        result.update(
            {
                "type": "multiplicity",
                "components": [{"name": low[2:], "count": 2}],
                "confidence": "medium",
                "needs_manual_review": True,
                "notes": "prefix_di",
            }
        )
        return result

    # mixtures / percentages
    if re.search(r"\d+%", low):
        result.update(
            {
                "type": "mixture",
                "components": [{"name": text, "count": 1}],
                "confidence": "low",
                "needs_manual_review": True,
                "notes": "contains_percentage",
            }
        )
        return result

    # polymeric
    if any(k in low for k in ["polystyrene", "polyisobutylene", "pib", "pam"]):
        result.update(
            {
                "type": "polymeric",
                "components": [{"name": text, "count": 1}],
                "confidence": "low",
                "needs_manual_review": True,
                "notes": "polymeric",
            }
        )
        return result

    # parenthetical candidates for OPSIN
    chunks = extract_parenthetical_chunks(text)
    candidates = [c for c in chunks if re.search(r"\d-[-\w]", c)]
    if candidates:
        result.update(
            {
                "type": "parenthetical_opsin",
                "components": [{"name": text, "count": 1}],
                "confidence": "medium",
                "needs_manual_review": True,
                "opsin_candidates": candidates,
                "notes": "opsin_candidate",
            }
        )
        return result

    # descriptive fallback
    if len(low.split()) >= 6 or len(low) > 50:
        result.update(
            {
                "type": "descriptive",
                "components": [{"name": text, "count": 1}],
                "confidence": "low",
                "needs_manual_review": True,
                "notes": "long_descriptive",
            }
        )
        return result

    result["components"] = [{"name": text, "count": 1}]
    return result


def normalize_polymer_name(name: str) -> str:
    """
    Minimal normalization for polymer names used in metadata matching.
    """
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def normalize_sidechain_name(name: str) -> str:
    """
    Normalize sidechain descriptors.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = name.replace(" ", "")
    mapping = {
        "methyl": "c1",
        "ethyl": "c2",
        "propyl": "c3",
        "butyl": "c4",
        "pentyl": "c5",
        "hexyl": "c6",
        "heptyl": "c7",
        "octyl": "c8",
        "nonyl": "c9",
        "decyl": "c10",
        "dodecyl": "c12",
        "octyldodecyl": "branched c20",
    }
    for k, v in mapping.items():
        if k in name:
            name = name.replace(k, v)
    name = re.sub(r"[^a-z0-9\-\s]", "", name)
    return name


def sanitize_smiles(smiles: str):
    """
    Single-pass SMILES cleanup and validation.
    Returns None on invalid/empty inputs.
    """
    if smiles is None or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    if Chem is None:
        return s
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def safe_apply_smiles(df, col: str, debug: bool = False):
    """
    Safely sanitize a SMILES column in a DataFrame.
    Adds a <col>_clean column with sanitized SMILES.
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if col not in df.columns:
        return df
    df = df.copy()
    try:
        df[f"{col}_clean"] = df[col].apply(lambda x: sanitize_smiles(x) if isinstance(x, str) else None)
    except Exception:
        df[f"{col}_clean"] = None
    return df


def parse_boolean(value) -> bool:
    """
    Robustly parse a value into a boolean.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "t", "yes", "y", "1"}:
            return True
        if val in {"false", "f", "no", "n", "0"}:
            return False
    return False


def sidechain_text_to_length(text: str) -> int:
    """
    Infer approximate alkyl chain length from sidechain description.
    """
    if not isinstance(text, str):
        return 0
    s = text.strip().lower()

    match = re.search(r"c(\\d{1,2})", s)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass

    mapping = {
        "methyl": 1,
        "ethyl": 2,
        "propyl": 3,
        "butyl": 4,
        "pentyl": 5,
        "hexyl": 6,
        "heptyl": 7,
        "octyl": 8,
        "nonyl": 9,
        "decyl": 10,
        "undecyl": 11,
        "dodecyl": 12,
        "tridecyl": 13,
        "tetradecyl": 14,
        "pentadecyl": 15,
        "hexadecyl": 16,
        "heptadecyl": 17,
        "octadecyl": 18,
        "nonadecyl": 19,
        "eicosyl": 20,
        "octyldodecyl": 20,
        "2-octyldodecyl": 20,
    }
    for k, v in mapping.items():
        if k in s:
            return v

    return 0


def alkyl_smiles_from_length(length: int) -> str:
    """
    Generate a simple linear alkyl SMILES for a given length.
    """
    try:
        length = int(length)
    except Exception:
        return ""
    if length < 1:
        return ""
    return "C" * length
