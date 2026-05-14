"""
Drug ID resolution and canonicalisation.

Converts human-readable drug names (e.g. "Aspirin", "aspirin repurposing")
to the stable, lower-case underscore IDs stored in ChromaDB metadata
(e.g. "aspirin").

Two lookup sources are supported:
- Filesystem scan (build_drug_lookup): reads PDF filenames under docs_dir.
- ChromaDB metadata scan (build_drug_lookup_from_metadatas): reads drug_id
  fields already stored in the vector store — preferred at runtime because
  it stays in sync with ingested data without requiring disk access.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Iterable, Any

from .utils import parse_filename


def normalize_drug_id_input(drug_name: str) -> str:
    """Strip whitespace and lowercase a raw drug name string."""
    return drug_name.strip().lower()


def _canonicalize(name: str) -> str:
    """
    Produce a maximally-reduced key for fuzzy drug name matching.

    Removes spaces, hyphens, underscores, the word 'repurposing', and all
    non-alphanumeric characters so that 'Aspirin Repurposing', 'aspirin',
    and 'aspirin-repurposing' all map to the same key ('aspirin').
    """
    name = name.strip().lower()
    if not name:
        return ""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\brepurposing\b", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def canonicalize_drug_name(name: str) -> str:
    """Public wrapper — returns the canonical key for a drug name string."""
    return _canonicalize(name)


def build_drug_lookup(docs_dir: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Build drug ID set and alias map by scanning PDF filenames on disk.

    Walks docs_dir recursively, parses each PDF filename, and registers the
    resulting drug_id plus the parent folder name as lookup aliases.

    Args:
        docs_dir: Root directory containing per-drug PDF subfolders.

    Returns:
        Tuple of (drug_ids, canonical_map) where canonical_map maps a
        canonicalised alias key to the set of matching drug IDs.
    """
    drug_ids: Set[str] = set()
    canonical_map: Dict[str, Set[str]] = {}

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return drug_ids, canonical_map

    for pdf_path in docs_path.rglob("*.pdf"):
        try:
            doc_info = parse_filename(pdf_path.name, pdf_path.parent.name)
        except ValueError:
            continue

        drug_id = doc_info.drug_id
        drug_ids.add(drug_id)

        for alias in {drug_id, pdf_path.parent.name}:
            key = _canonicalize(alias)
            if not key:
                continue
            canonical_map.setdefault(key, set()).add(drug_id)

    return drug_ids, canonical_map


def build_drug_lookup_from_metadatas(
    metadatas: Iterable[Dict[str, Any]]
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Build drug ID set and alias map from ChromaDB chunk metadata.

    Preferred over build_drug_lookup at runtime because it reflects the actual
    ingested state rather than the filesystem.

    Args:
        metadatas: Iterable of metadata dicts, each expected to contain
                   a 'drug_id' key.

    Returns:
        Tuple of (drug_ids, canonical_map).
    """
    drug_ids: Set[str] = set()
    canonical_map: Dict[str, Set[str]] = {}

    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        drug_id = (meta.get("drug_id") or "").strip().lower()
        if not drug_id:
            continue
        drug_ids.add(drug_id)

        # Register underscore, space, and hyphen variants as aliases
        aliases = {drug_id, drug_id.replace("_", " "), drug_id.replace("_", "-")}
        for alias in aliases:
            key = _canonicalize(alias)
            if not key:
                continue
            canonical_map.setdefault(key, set()).add(drug_id)

    return drug_ids, canonical_map


def resolve_drug_id(
    drug_name: str,
    drug_ids: Set[str],
    canonical_map: Dict[str, Set[str]],
    allow_fallback: bool = True
) -> Optional[str]:
    """
    Resolve a human-readable drug name to its canonical drug ID.

    Resolution order:
    1. Exact match in drug_ids (after lower-casing).
    2. Single unambiguous match via canonical_map.
    3. If multiple candidates exist and the normalised name is one of them, use it.
    4. Fallback: return the normalised name as-is (useful when drug_ids is empty
       and ChromaDB filtering should handle unknown names gracefully).

    Args:
        drug_name: Raw input string from the user or API caller.
        drug_ids: Known set of canonical drug IDs.
        canonical_map: Alias-to-drug-ID mapping from build_drug_lookup*.
        allow_fallback: When True, return the normalised input even if it is not
                        in drug_ids (avoids 404 when data hasn't been indexed yet).

    Returns:
        Canonical drug ID string, or None if unresolvable.
    """
    if not drug_name:
        return None

    normalized = normalize_drug_id_input(drug_name)
    if normalized in drug_ids:
        return normalized

    canonical = _canonicalize(normalized)
    candidates = canonical_map.get(canonical, set())
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1 and normalized in candidates:
        return normalized

    if allow_fallback:
        return normalized or None

    return None
