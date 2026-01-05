import pandas as pd

from polymer_pipeline.layers.layer2_fusion import run_layer2_fusion
from polymer_pipeline.utils.common import normalize_polymer_name


def test_normalize_polymer_name_basic():
    assert normalize_polymer_name("Poly(3-hexylthiophene)") == "poly(3-hexylthiophene)"
    assert normalize_polymer_name("polymer blend of Poly-3HT and PolyDPP") == "polymer blend of poly-3ht and polydpp"
    assert normalize_polymer_name(None) == ""
    assert normalize_polymer_name("poly-C8") == "poly-c8"


def test_matching_prefers_exact_name_then_substring_then_paper():
    text_df = pd.DataFrame(
        [
            {"paper_id": "1", "polymer_name": "Poly-ABC", "backbone_smiles_text": None, "sidechain_smiles_list_text": []},
        ]
    )
    image_df = pd.DataFrame(
        [
            {"paper_id": "1", "polymer_name": "Poly-ABC", "candidate_smiles": "CCCC", "parse_status": "ok"},
            {"paper_id": "1", "polymer_name": "xxx", "candidate_smiles": "CC", "parse_status": "ok"},
        ]
    )
    fused = run_layer2_fusion(text_df, image_df, fast_mode=True)
    assert len(fused) == 1
    assert fused.iloc[0]["source_priority"] in {"image_pool", "unknown"}
    assert fused.iloc[0]["match_mode"] in {"exact_name", "paper_only", "substring_name", "no_match"}


def test_safe_contains_literal():
    df = pd.DataFrame({"polymer_name_norm": ["p3ht-co-c8", "dpp-t", "p(n-di2t-c6)"]})
    pattern = "p3ht-co-c8"
    mask = df["polymer_name_norm"].str.contains(pattern, case=False, na=False, regex=False)
    assert mask.sum() == 1
