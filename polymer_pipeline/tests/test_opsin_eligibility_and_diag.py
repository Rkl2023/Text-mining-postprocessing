import pytest

import polymer_pipeline.layers.layer1_parse_text as lpt
from polymer_pipeline.utils import opsin_bridge


def test_attempt_opsin_diag_shapes(monkeypatch):
    """_attempt_opsin returns legacy single value or tuple with diag when requested."""
    def fake_parse_with_diag(name):
        return "CC", {"cached": False, "status": "SUCCESS", "message": "", "warnings": [], "exception": None}

    monkeypatch.setattr(opsin_bridge, "parse_chemical_name_with_diag", fake_parse_with_diag)
    smi_only = lpt._attempt_opsin("ethanol")
    assert smi_only in (None, "CC") or isinstance(smi_only, str)

    smi, diag = lpt._attempt_opsin("ethanol", with_diag=True)
    assert isinstance(diag, dict)
    assert smi in (None, "CC")


def test_ineligible_backbone_skips_opsin(monkeypatch):
    """Descriptive backbone is classified ineligible and should not call OPSIN."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("OPSIN should not be called for ineligible text")

    monkeypatch.setattr(lpt, "_attempt_opsin", fail_if_called)
    smi, source, conf, traces = lpt._parse_backbone_with_stages("BDT-based D–A copolymer", fast_mode=True)
    assert smi is None
    assert source == "TEXT_FALLBACK"
    assert any(t["stage"] == "opsin_skip" for t in traces)


def test_eligible_backbone_attempts_opsin(monkeypatch):
    """Simple OPSIN-parseable name remains eligible and attempted."""
    def fake_attempt(name, with_diag=False):
        if with_diag:
            return "C", {"status": "SUCCESS"}
        return "C"

    monkeypatch.setattr(lpt, "_attempt_opsin", fake_attempt)
    smi, source, conf, traces = lpt._parse_backbone_with_stages("benzothiadiazole", fast_mode=True)
    assert smi == "C"
    assert any(t["stage"] == "opsin_direct" for t in traces)
    assert source == "OPSIN"
    assert conf == "high"

