import pytest

from polymer_pipeline.utils.common import _fix_punctuation_errors, _normalize_sidechain_tokens
from polymer_pipeline.layers.layer1_parse_text import preprocess_opsin_name, _parse_sidechain_with_trace


def test_fix_punctuation_errors_normalizes_and_trims():
    text = "Thiophene—,,; "
    fixed = _fix_punctuation_errors(text)
    assert fixed == "Thiophene-"


def test_preprocess_opsin_name_applies_locant_and_diyl():
    raw = "Thiophene and benzodithiophene polymer"
    preprocessed, flags, logs = preprocess_opsin_name(raw, kind="backbone")
    assert "thiophene-2,5-diyl" in preprocessed
    assert "benzodithiophene-2,6-diyl" in preprocessed
    assert flags["locant_augmented"] is True
    assert flags["diyl_appended"] is True
    assert any("[OPSIN-PREPROC]" in note for note in logs)


def test_sidechain_preprocess_in_trace():
    smi, cleaned, source, success, traces = _parse_sidechain_with_trace("thiophene conjugated sidechain")
    assert cleaned.startswith("thiophene")
    assert any(t.get("stage") == "sidechain_preprocessed_opsin" for t in traces)
    assert any("preprocessed_for_opsin" in t for t in traces)


def test_sidechain_dictionary_expansion_doc10():
    preprocessed, flags, _ = preprocess_opsin_name("doc10", kind="sidechain")
    assert "2-decyldodecyl" in preprocessed
    assert flags["sidechain_token_expanded"] is True


def test_sidechain_c8oxy_normalization():
    normalized, flags = _normalize_sidechain_tokens("c8oxy")
    assert "octyl" in normalized
    assert "oxy" in normalized
    assert flags["tokens_normalized"] is True
