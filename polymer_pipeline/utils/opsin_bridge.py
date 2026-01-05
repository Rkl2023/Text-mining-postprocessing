import atexit
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    import jpype  # type: ignore
    from jpype import JClass  # type: ignore

    JPYPE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    jpype = None
    JClass = None
    JPYPE_AVAILABLE = False
    _JPYPE_IMPORT_ERROR = exc
else:
    _JPYPE_IMPORT_ERROR = None

# Local, prebuilt OPSIN jar (Java backend only)
OPSIN_JAR_PATH = Path("/Users/k25063738/Text-mining-postprocessing/OPSIN/opsin-cli/target/opsin-cli-3.0-SNAPSHOT-jar-with-dependencies.jar")
OPSIN_DIR = Path(os.path.expanduser("~/.opsin"))
OPSIN_CACHE = OPSIN_DIR / "cache.json"
OPSIN_VERSION = "3.0-SNAPSHOT"
JVM_PATH = "/opt/homebrew/Cellar/openjdk/25.0.1/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"
OPSIN_PACKAGE = None

logger = logging.getLogger("polymer_pipeline")

_opsin_instance = None
_opsin_cache = None  # persisted on disk
_opsin_cache_mem = {}  # fast in-memory cache
_opsin_cache_dirty = False
_opsin_init_attempted = False
OPSIN_AVAILABLE = False
_jvm_error: Optional[str] = None
_jvm_path: Optional[str] = None
# Simple counters for instrumentation
OPSIN_CALLS = 0
OPSIN_CACHE_HITS = 0
OPSIN_CACHE_MISSES = 0
OPSIN_SUCCESS = 0


def _load_cache():
    global _opsin_cache, _opsin_cache_mem
    if _opsin_cache is not None:
        return
    try:
        if OPSIN_CACHE.exists():
            _opsin_cache = json.loads(OPSIN_CACHE.read_text())
        else:
            _opsin_cache = {}
    except Exception:
        _opsin_cache = {}
    _opsin_cache_mem = dict(_opsin_cache)


def _save_cache():
    global _opsin_cache_dirty
    if not _opsin_cache_dirty or _opsin_cache is None:
        return
    try:
        OPSIN_DIR.mkdir(parents=True, exist_ok=True)
        OPSIN_CACHE.write_text(json.dumps(_opsin_cache, indent=2))
        _opsin_cache_dirty = False
    except Exception:
        # Best-effort persistence; keep running even if flush fails.
        pass


def _flush_cache():
    """Flush OPSIN cache to disk once at process exit."""
    try:
        _save_cache()
    except Exception:
        pass


atexit.register(_flush_cache)


def ensure_opsin():
    """
    Ensure OPSIN JVM is initialized using the local 3.0-SNAPSHOT jar.
    """
    global _opsin_instance, OPSIN_AVAILABLE, _opsin_init_attempted, _jvm_error, _jvm_path, OPSIN_PACKAGE
    if _opsin_instance is not None:
        return _opsin_instance
    if _opsin_init_attempted:
        return _opsin_instance
    _opsin_init_attempted = True

    if not JPYPE_AVAILABLE:
        logger.info("OPSIN bridge disabled — install jpype1>=1.5.0 to enable JVM parsing (%s)", _JPYPE_IMPORT_ERROR)
        OPSIN_AVAILABLE = False
        return None

    if not OPSIN_JAR_PATH.exists():
        logger.warning("OPSIN jar not found at %s; please build or place the 3.0-SNAPSHOT jar locally.", OPSIN_JAR_PATH)
        OPSIN_AVAILABLE = False
        return None

    if not jpype.isJVMStarted():
        try:
            _jvm_path = JVM_PATH
            logger.info("Attempting to start JVM at %s", _jvm_path)
            t0 = time.perf_counter()
            jpype.startJVM(
                _jvm_path,
                "-ea",
                f"-Djava.class.path={OPSIN_JAR_PATH}",
                convertStrings=True,
            )
            logger.info("[OPSIN] JVM started successfully using %s (jar=%s) in %.2fs", _jvm_path, OPSIN_JAR_PATH, time.perf_counter() - t0)
        except Exception as exc:
            _jvm_error = str(exc)
            logger.warning("[OPSIN] JVM startup failed: %s", exc)
            OPSIN_AVAILABLE = False
            return None

    try:
        pkg_candidates = ["uk.ac.cam.ch.wwmm.opsin.NameToStructure", "uk.ac.cam.ch.opsin.NameToStructure"]
        for candidate in pkg_candidates:
            try:
                cls = JClass(candidate)
                _opsin_instance = cls.getInstance()
                if _opsin_instance:
                    OPSIN_PACKAGE = candidate
                    OPSIN_AVAILABLE = True
                    logger.info("[OPSIN] Using package: %s", OPSIN_PACKAGE)
                    break
            except Exception:
                continue
        if not _opsin_instance:
            logger.warning("[OPSIN] Could not find NameToStructure class in jar.")
            OPSIN_AVAILABLE = False
            return None
    except Exception as exc:
        logger.warning("Failed to instantiate OPSIN: %s", exc)
        _opsin_instance = None
        OPSIN_AVAILABLE = False
    return _opsin_instance


def _parse_chemical_name(name: str, return_diag: bool = False):
    """
    Parse chemical name to SMILES using OPSIN, returns canonical SMILES or None.
    Uses in-memory cache for speed; disk cache flushes once per run.
    """
    global _opsin_instance, _opsin_cache, _opsin_cache_mem, OPSIN_CALLS, OPSIN_CACHE_HITS, OPSIN_CACHE_MISSES, OPSIN_SUCCESS, _opsin_cache_dirty
    diag = {
        "cached": False,
        "status": None,
        "message": None,
        "warnings": None,
        "exception": None,
        "elapsed": None,
    }
    if not isinstance(name, str) or not name.strip():
        return (None, diag) if return_diag else None
    name = name.strip()
    _load_cache()
    if name in _opsin_cache_mem:
        OPSIN_CACHE_HITS += 1
        diag["cached"] = True
        diag["status"] = "CACHE_HIT"
        return (_opsin_cache_mem[name], diag) if return_diag else _opsin_cache_mem[name]

    OPSIN_CALLS += 1
    OPSIN_CACHE_MISSES += 1

    if _opsin_instance is None:
        _opsin_instance = ensure_opsin()
    if _opsin_instance is None:
        logger.warning("[OPSIN] Java backend unavailable — skipping OPSIN parsing.")
        diag["status"] = "JVM_UNAVAILABLE"
        return (None, diag) if return_diag else None

    try:
        t0 = time.perf_counter()
        result = _opsin_instance.parseChemicalName(name)
        elapsed = time.perf_counter() - t0
        diag["elapsed"] = elapsed
        smiles = result.getSmiles() if result else None
        try:
            if result:
                diag["status"] = str(result.getStatus()) if hasattr(result, "getStatus") else None
                diag["message"] = result.getMessage() if hasattr(result, "getMessage") else None
                if hasattr(result, "getWarnings"):
                    warnings = result.getWarnings()
                    if warnings is not None:
                        diag["warnings"] = len(warnings)
        except Exception as exc:  # pragma: no cover - best effort
            diag["exception"] = f"diag_error:{exc}"
        logger.info("OPSIN parsed '%s' in %.3fs", name, time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("OPSIN parsing failed for '%s': %s", name, exc)
        diag["exception"] = f"{exc.__class__.__name__}: {exc}"
        smiles = None

    if smiles and _opsin_cache is not None:
        _opsin_cache_mem[name] = smiles
        _opsin_cache[name] = smiles
        _opsin_cache_dirty = True
        OPSIN_SUCCESS += 1
    return (smiles, diag) if return_diag else smiles


@lru_cache(maxsize=10000)
def parse_chemical_name(name: str) -> Optional[str]:
    """LRU-wrapped public API."""
    return _parse_chemical_name(name)


def parse_chemical_name_with_diag(name: str):
    """
    Backward-compatible helper that returns (smiles, diag).
    Does not change caching semantics for successes.
    """
    return _parse_chemical_name(name, return_diag=True)


def get_opsin_version() -> str:
    """Return the configured OPSIN backend version string."""
    if OPSIN_PACKAGE == "uk.ac.cam.ch.wwmm.opsin.NameToStructure":
        return f"{OPSIN_VERSION} (wwmm)"
    return OPSIN_VERSION


def get_jvm_status():
    """Report JVM startup state for diagnostics."""
    return {
        "started": jpype.isJVMStarted() if JPYPE_AVAILABLE else False,
        "jvm_path": _jvm_path,
        "error": _jvm_error,
    }


def get_opsin_status():
    """
    Lightweight status helper for UI diagnostics.
    """
    return {
        "java_opsin": OPSIN_AVAILABLE,
        "jpype": JPYPE_AVAILABLE,
        "jar_path": str(OPSIN_JAR_PATH),
        "version": OPSIN_VERSION,
        "package": OPSIN_PACKAGE,
    }


def combine_opsin_template_fragments(template_smiles, opsin_smiles_list):
    """
    Merge template-derived SMILES with OPSIN-derived fragments.
    """
    fragments = []
    if template_smiles:
        if isinstance(template_smiles, (list, tuple, set)):
            fragments.extend([s for s in template_smiles if s])
        else:
            fragments.append(template_smiles)
    if opsin_smiles_list:
        fragments.extend([s for s in opsin_smiles_list if s])
    fragments = [s for s in fragments if s]
    if not fragments:
        return None
    joined = ".".join(fragments)
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(joined, sanitize=True)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return joined
