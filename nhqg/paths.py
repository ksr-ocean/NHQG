from __future__ import annotations

from pathlib import Path


OUTPUT_ROOT = "output"


def output_root() -> Path:
    return Path(OUTPUT_ROOT)


def normalize_output_dir(path: str | Path) -> str:
    """Place relative output paths under the repo-level output/ tree."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if not p.parts:
        return str(output_root())
    if p.parts[0] in {OUTPUT_ROOT, ".", ".."}:
        return str(p)
    return str(output_root() / p)


def resolve_existing_output_path(path: str | Path) -> Path:
    """Resolve existing archives under output/ while tolerating absolute paths."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    if not p.parts or p.parts[0] in {OUTPUT_ROOT, ".", ".."}:
        return p
    candidate = output_root() / p
    if candidate.exists():
        return candidate
    return p
