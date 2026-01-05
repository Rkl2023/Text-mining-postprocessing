import logging
from typing import Dict, List, Optional

import pandas as pd

from .utils import alkyl_smiles_from_length, sanitize_smiles, sidechain_text_to_length

logger = logging.getLogger("polymer_pipeline")

# Common conjugated backbone templates used in OPV/OECT literature.
# Expanded to cover many frequent motifs; SMILES are simplified scaffolds.
BACKBONE_DICTIONARY: Dict[str, str] = {
    "bdt": "c1sc(cc1)c2sc(cc2)",
    "benzodithiophene": "c1sc(cc1)c2sc(cc2)",
    "dpp": "N1C(=O)C(C(=O)N1)c2ccc(cc2)c3ccc(cc3)",
    "diketopyrrolopyrrole": "N1C(=O)C(C(=O)N1)c2ccc(cc2)c3ccc(cc3)",
    "ndi": "O=C1Nc2ccc(cc2C(=O)N1)c3ccc4ccccc4c3",
    "naphthalene diimide": "O=C1Nc2ccc(cc2C(=O)N1)c3ccc4ccccc4c3",
    "iid": "O=C(Nc1ccc(cc1))Nc2ccc(cc2)C(=O)",
    "isoindigo": "O=C(Nc1ccc(cc1))Nc2ccc(cc2)C(=O)",
    "tt": "c1sc2sccc2s1",
    "thienothiophene": "c1sc2sccc2s1",
    "tvt": "c1sc2sc3ccccc3s2s1",
    "pbdb-t": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pm6": "c1sc(cc1)c2cc(cc(c2)C)C",
    "y6": "c1cc(c(c(c1)C(=O)O)C(=O)O)C",
    "p3ht": "c1sc(cc1)C",
    "p3bt": "c1sc(cc1)C",
    "ptb7": "c1sc(cc1)c2cc(cc(c2)C)C",
    "ptb7-th": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pffbt4t": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pffbt4t-2od": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pdcbt": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pidt": "c1sc2cccc3ccc(sc23)c1",
    "pidt-tt": "c1sc2cccc3ccc(sc23)c1",
    "pidtt": "c1sc2cccc3ccc(sc23)c1",
    "pce10": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pce13": "c1sc(cc1)c2cc(cc(c2)C)C",
    "pbttt": "c1sc(cc1)c2cc(cc(c2)C)C",
    "p(ndi2od-t2)": "O=C1Nc2ccc(cc2C(=O)N1)c3ccc4ccccc4c3",
    "pdi": "O=C1Nc2ccc(cc2C(=O)N1)c3ccc4ccccc4c3",
    "ppv": "c1ccccc1",
    "pfdt2bz": "c1sc(cc1)c2sc(cc2)",
    "psbtbt": "c1sc(cc1)c2sc(cc2)",
    "ptcb": "c1sc(cc1)c2sc(cc2)",
    "pqt": "c1sc(cc1)c2sc(cc2)",
    "pqdt": "c1sc(cc1)c2sc(cc2)",
    "pbbt": "c1sc(cc1)c2sc(cc2)",
    "pbtp": "c1sc(cc1)c2sc(cc2)",
    "pbbdt": "c1sc(cc1)c2sc(cc2)",
    "pb2t": "c1sc(cc1)c2sc(cc2)",
    "pfdtbt": "c1sc(cc1)c2sc(cc2)",
    "pdcqp": "c1sc(cc1)c2sc(cc2)",
    "pbqp": "c1sc(cc1)c2sc(cc2)",
    "pdppt": "c1sc(cc1)c2sc(cc2)",
    "pndit2": "O=C1Nc2ccc(cc2C(=O)N1)c3ccc4ccccc4c3",
    "pn-se": "c1sc(cc1)c2sc(cc2)",
    "pbbdttt": "c1sc(cc1)c2sc(cc2)",
    "pcdtbt": "c1sc(cc1)c2sc(cc2)",
    "pbdttt": "c1sc(cc1)c2sc(cc2)",
    "pbdttpd": "c1sc(cc1)c2sc(cc2)",
    "pbtaz": "c1sc(cc1)c2sc(cc2)",
    "pcbt": "c1sc(cc1)c2sc(cc2)",
    "pbnbt": "c1sc(cc1)c2sc(cc2)",
    "pbdtts": "c1sc(cc1)c2sc(cc2)",
    "pbdt-ttpd": "c1sc(cc1)c2sc(cc2)",
    "pbdtt-se": "c1sc(cc1)c2sc(cc2)",
    "pquaterthiophene": "c1sc(cc1)c2sc(cc2)",
    "psexithiophene": "c1sc(cc1)c2sc(cc2)",
}

SIDECHAIN_DICTIONARY: Dict[str, str] = {
    "c6": "hexyl",
    "c8": "octyl",
    "c10": "decyl",
    "doc10": "2-decyldodecyl",
    "oxy": "oxy",
    "alkoxy": "alkoxy",
    "eg": "ethoxyethyl",
    "peg": "poly(ethylene glycol)",
    "thio": "thio",
    "siloxane": "siloxane",
    "phenyl": "phenyl",
}


def name_to_smiles_via_opsin(name: str) -> Optional[str]:
    if not name:
        return None
    try:
        from polymer_pipeline.utils.opsin_bridge import parse_chemical_name
    except Exception as exc:
        logger.debug("OPSIN bridge unavailable: %s", exc)
        return None

    try:
        smiles = parse_chemical_name(name)
        return sanitize_smiles(smiles) if smiles else None
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("OPSIN failed for '%s': %s", name, exc)
        return None


def backbone_from_name(text: str) -> Optional[str]:
    if not text:
        return None
    text_lower = text.lower()
    for key, smiles in BACKBONE_DICTIONARY.items():
        if key in text_lower:
            return sanitize_smiles(smiles)
    return name_to_smiles_via_opsin(text)


def sidechain_from_name(text: str) -> Optional[str]:
    """
    Parse side-chain descriptors, prioritizing alkyl chain length.
    """
    if not text:
        return None
    length = sidechain_text_to_length(text)
    if length:
        return sanitize_smiles(alkyl_smiles_from_length(length))
    return name_to_smiles_via_opsin(text)


def run_name_to_structure(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1: Convert backbone and side-chain textual descriptors to SMILES.
    """
    df = text_df.copy()
    backbone_smiles: List[Optional[str]] = []
    sidechain_smiles_list: List[List[str]] = []

    for _, row in df.iterrows():
        backbone_name = row.get("backbone_name_text", "")
        sidechain_name = row.get("sidechain_name_text_norm", row.get("sidechain_name_text", ""))

        backbone = backbone_from_name(backbone_name)
        sidechain = sidechain_from_name(sidechain_name)
        backbone_smiles.append(backbone)

        sidechains: List[str] = []
        if sidechain:
            sidechains.append(sidechain)
        df_sidechain_col = row.get("sidechain_smiles_list")
        if isinstance(df_sidechain_col, list):
            sidechains.extend([s for s in df_sidechain_col if s])
        sidechain_smiles_list.append(list(dict.fromkeys(sidechains)))  # deduplicate preserve order

    df["backbone_smiles_text"] = backbone_smiles
    df["sidechain_smiles_list_text"] = sidechain_smiles_list
    logger.info(
        "Layer 1 complete: %d/%d entries yielded backbone SMILES from text, %d had sidechains",
        sum(1 for b in backbone_smiles if b),
        len(df),
        sum(1 for lst in sidechain_smiles_list if lst),
    )
    return df
