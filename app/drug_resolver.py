import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from .utils import parse_filename


def normalize_drug_id_input(drug_name: str) -> str:
    return drug_name.strip().lower()


def _canonicalize(name: str) -> str:
    name = name.strip().lower()
    if not name:
        return ""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\brepurposing\b", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def build_drug_lookup(docs_dir: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
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


def resolve_drug_id(
    drug_name: str,
    drug_ids: Set[str],
    canonical_map: Dict[str, Set[str]],
    allow_fallback: bool = True
) -> Optional[str]:
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
