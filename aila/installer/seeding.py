from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

MODE_600 = 0o600


@dataclass(frozen=True)
class SeedReport:
    created: tuple[Path, ...]
    skipped: tuple[Path, ...]


def ensure_mode_600(path: str | Path) -> None:
    Path(path).chmod(MODE_600)


def write_text_if_absent(
    target_path: str | Path,
    text: str,
    *,
    mode: int | None = None,
) -> bool:
    """Write text only when target does not already exist."""
    target = Path(target_path)
    if target.exists():
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if mode is not None:
        target.chmod(mode)
    return True


def seed_file_if_absent(
    source_path: str | Path,
    target_path: str | Path,
    *,
    mode: int | None = None,
) -> bool:
    """Copy a seed file only when target does not already exist."""
    source = Path(source_path)
    target = Path(target_path)
    if target.exists():
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if mode is not None:
        target.chmod(mode)
    return True


copy_file_if_absent = seed_file_if_absent
create_file_if_absent = write_text_if_absent


def seed_tree_if_absent(
    source_root: str | Path,
    target_root: str | Path,
    *,
    file_modes: Mapping[str, int] | None = None,
) -> SeedReport:
    """Copy a seed tree without overwriting any existing target files."""
    source = Path(source_root)
    target = Path(target_root)
    modes = dict(file_modes or {})
    created: list[Path] = []
    skipped: list[Path] = []

    for source_path in sorted(source.rglob("*")):
        relative_path = source_path.relative_to(source)
        target_path = target / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        mode = modes.get(relative_path.as_posix())
        if seed_file_if_absent(source_path, target_path, mode=mode):
            created.append(target_path)
        else:
            skipped.append(target_path)

    return SeedReport(created=tuple(created), skipped=tuple(skipped))
