import polymer_pipeline.layers.layer1_parse_text as lpt
import polymer_pipeline.utils.common as common


def test_decoder_mapping_branch():
    decoded = common.decode_sidechain_raw("2-octyldodecyl")
    assert decoded["smiles"] is not None
    assert decoded["confidence"] == "high"


def test_formula_parser():
    decoded = common.decode_sidechain_raw("n-C8H17")
    assert decoded["smiles"] is not None
    assert decoded["type"] == "formula_linear"


def test_multiplicity_metadata():
    decoded = common.decode_sidechain_raw("two hexyl chains")
    assert decoded["type"] == "multiplicity"
    assert decoded["smiles"] is None


def test_parenthetical_opsin_candidate():
    txt = "phosphonate (bis(2-ethylhexyl)(4-bromobutyl)phosphonate)"
    decoded = common.decode_sidechain_raw(txt)
    assert decoded["opsin_candidates"]


def test_polymeric_detection():
    decoded = common.decode_sidechain_raw("atactic polystyrene (Mn = 1300 g/mol) and 2-octyldodecyl")
    assert decoded["type"] == "polymeric" or decoded["needs_manual_review"]


def test_gentle_opsin_prep_respects_text():
    assert common.preprocess_opsin_name_sidechain_gentle("2-ethylhexyl") == "2-ethylhexyl"

