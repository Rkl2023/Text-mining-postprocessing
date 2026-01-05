import logging

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    Chem = None
    Descriptors = None
    MurckoScaffold = None
logger = logging.getLogger("polymer_pipeline")


def ensure_rdkit_import():
    """
    Attempt a late import of RDKit if it was unavailable at module load time.
    Returns the Chem module (or None if still unavailable).
    """
    global Chem, RDLogger, Descriptors, MurckoScaffold
    if Chem is not None:
        return Chem
    try:
        from rdkit import Chem as _Chem, RDLogger as _RDLogger
        from rdkit.Chem import Descriptors as _Descriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold as _MurckoScaffold

        _RDLogger.DisableLog("rdApp.error")
        _RDLogger.DisableLog("rdApp.warning")

        Chem = _Chem
        RDLogger = _RDLogger
        Descriptors = _Descriptors
        MurckoScaffold = _MurckoScaffold
    except Exception:
        Chem = None
        Descriptors = None
        MurckoScaffold = None
    return Chem


def is_valid_smiles(smiles: str) -> bool:
    """
    Validate whether a SMILES string can be parsed by RDKit.
    """
    if not isinstance(smiles, str):
        return False
    if Chem is None:
        return bool(smiles.strip())
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert arbitrary SMILES into canonical RDKit format.
    """
    if not isinstance(smiles, str):
        return ""
    if Chem is None:
        return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def get_molecular_weight(smiles: str) -> float:
    """
    Compute molecular weight (g/mol) for a valid SMILES string.
    """
    if Chem is None or Descriptors is None:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0
        return Descriptors.MolWt(mol)
    except Exception:
        return 0.0


def repair_smiles(smiles: str) -> str:
    """
    Attempt minimal SMILES repair via cleanup and RDKit parsing.
    """
    if not isinstance(smiles, str):
        return ""
    if Chem is None:
        return smiles.strip()
    s = smiles.strip()
    s = s.replace("＝", "=").replace("–", "-").replace("—", "-")
    s = s.replace(" ", "").replace(".", "")
    try:
        mol = Chem.MolFromSmiles(s)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return ""
    except Exception:
        return ""


def smiles_summary(smiles: str) -> dict:
    """
    Return a summary dictionary for valid SMILES.
    """
    summary = {"valid": False, "canonical": "", "mw": 0.0, "num_atoms": 0}
    if Chem is None or Descriptors is None:
        return summary
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            summary["valid"] = True
            summary["canonical"] = Chem.MolToSmiles(mol, canonical=True)
            summary["mw"] = Descriptors.MolWt(mol)
            summary["num_atoms"] = mol.GetNumAtoms()
    except Exception:
        pass
    return summary


def extract_longest_aliphatic_chain(smiles: str) -> str:
    """
    Extract the longest contiguous aliphatic carbon chain fragment.
    """
    if not isinstance(smiles, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        longest_chain = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            stack = [(atom, [atom.GetIdx()])]
            while stack:
                current_atom, path = stack.pop()
                for nbr in current_atom.GetNeighbors():
                    if nbr.GetSymbol() == "C" and nbr.GetIdx() not in path:
                        stack.append((nbr, path + [nbr.GetIdx()]))
                if len(path) > len(longest_chain):
                    longest_chain = path
        if not longest_chain:
            return ""
        submol = Chem.PathToSubmol(mol, longest_chain)
        return Chem.MolToSmiles(submol, canonical=True)
    except Exception:
        return ""


def count_aliphatic_chain_length(smiles: str, mol=None) -> int:
    """
    Return the length of the longest aliphatic chain in a SMILES.
    """
    frag = extract_longest_aliphatic_chain(smiles) if mol is None else extract_longest_aliphatic_chain(Chem.MolToSmiles(mol, canonical=True) if Chem else smiles)
    if not frag:
        return 0
    try:
        mol_local = Chem.MolFromSmiles(frag) if mol is None else Chem.MolFromSmiles(frag)
        return mol_local.GetNumAtoms() if mol_local else 0
    except Exception:
        return 0


def murcko_scaffold_smiles(smiles: str, include_chirality: bool = False, mol=None) -> str:
    """
    Return Bemis–Murcko scaffold SMILES for a molecule.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    if Chem is None or MurckoScaffold is None:
        return ""
    try:
        import time

        mol_local = mol or Chem.MolFromSmiles(smiles)
        if mol_local is None:
            return ""
        t0 = time.perf_counter()
        core = MurckoScaffold.GetScaffoldForMol(mol_local)
        dt = time.perf_counter() - t0
        if dt > 2.0:
            logger.warning("Murcko scaffold computation slow (%.2fs) for SMILES len=%d", dt, len(smiles))
        if core is None:
            return ""
        return Chem.MolToSmiles(core, canonical=True, includeChirality=include_chirality)
    except Exception:
        return ""


def murcko_framework_similarity(smiles_a: str, smiles_b: str) -> float:
    """
    Compute scaffold similarity between two SMILES using Tanimoto of Murcko fingerprints.
    """
    if Chem is None:
        return 0.0
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    try:
        sa = murcko_scaffold_smiles(smiles_a)
        sb = murcko_scaffold_smiles(smiles_b)
        if not sa or not sb:
            return 0.0
        ma = Chem.MolFromSmiles(sa)
        mb = Chem.MolFromSmiles(sb)
        if not ma or not mb:
            return 0.0
        fpa = FingerprintMols.FingerprintMol(ma)
        fpb = FingerprintMols.FingerprintMol(mb)
        return DataStructs.TanimotoSimilarity(fpa, fpb)
    except Exception:
        return 0.0
