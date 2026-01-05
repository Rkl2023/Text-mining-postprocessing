import polymer_pipeline.layers.layer1_parse_text as lpt
import polymer_pipeline.utils.common as common


def test_gentle_sidechain_no_yl_append():
    assert common.preprocess_opsin_name_sidechain_gentle("2-ethylhexyl") == "2-ethylhexyl"


def test_ineligible_conjugated_skips_opsin(monkeypatch):
    # Ensure OPSIN is not called for descriptive strings
    def fail_if_called(*args, **kwargs):
        raise AssertionError("OPSIN should not be called")

    monkeypatch.setattr(lpt, "_attempt_opsin", fail_if_called)
    smi, cleaned, source, success, traces = lpt._parse_sidechain_with_trace("ester-substitutedthienylconjugatedsidechains")
    assert source in {"NON_OPSIN_COMPATIBLE", "ALKYL_LENGTH", ""}
    assert any(t.get("stage") == "sidechain_non_opsin_compatible" for t in traces) or not success


def test_heuristic_unchanged(monkeypatch):
    smi1, cleaned1, source1, success1, traces1 = lpt._parse_sidechain_with_trace("c6")
    smi2, cleaned2, source2, success2, traces2 = lpt._parse_sidechain_with_trace("hexyl")
    assert smi1 == smi2 or (smi1 and smi2)
    assert source1 == "ALKYL_LENGTH"
    assert source2 == "ALKYL_LENGTH"
