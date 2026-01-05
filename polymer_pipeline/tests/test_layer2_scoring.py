import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - RDKit may be missing
    Chem = None

from polymer_pipeline.layer2_candidate_matching import _score_candidate


def _build_series(smiles: str):
    canonical = smiles
    heavy = 0
    if Chem:
        mol = Chem.MolFromSmiles(smiles)
        heavy = mol.GetNumHeavyAtoms() if mol else 0
    return pd.Series(
        {
            "canonical_smiles": canonical,
            "candidate_smiles_clean": canonical,
            "parse_status": "",
            "heavy_atom_count": heavy,
        }
    )


def test_aromatic_polymer_scores_positive():
    smi = "*Oc1nc2cc(Br)ccc2c2c(O*)nc3cc(Br)ccc3c12"
    row = _build_series(smi)
    score = _score_candidate(row, expected_chain_len=None, expect_eg=False, expect_ionic=False)
    assert score >= 2  # parsed_ok due to canonical_smiles present


def test_alkyl_polymer_no_regression():
    smi = "*CCCC(CCCCCCCCC)CCCCCCCCC"
    row = _build_series(smi)
    score = _score_candidate(row, expected_chain_len=None, expect_eg=False, expect_ionic=False)
    assert score >= 2
