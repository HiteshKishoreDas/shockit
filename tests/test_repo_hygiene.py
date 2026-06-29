from dataclasses import fields
import os
from pathlib import Path

from shockit import ShockFinderConfig


IGNORED_TOP_LEVEL_DIRS = {".git", ".venv", ".codex", ".agents", "test_artifacts", ".pytest_cache"}
TEXT_FILE_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg"}
TEXT_FILE_NAMES = {"LICENSE"}


def _iter_repo_text_files():
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in IGNORED_TOP_LEVEL_DIRS and directory != "__pycache__"
        ]
        for filename in files:
            path = Path(root) / filename
            if path == Path("tests/test_repo_hygiene.py"):
                continue
            if path.suffix not in TEXT_FILE_SUFFIXES and path.name not in TEXT_FILE_NAMES:
                continue
            yield path


def test_no_root_prompt_md() -> None:
    assert not Path("prompt.md").exists()


def test_repository_contains_no_absolute_user_paths() -> None:
    matches: list[str] = []
    for path in _iter_repo_text_files():
        text = path.read_text()
        if "/Users/" in text:
            matches.append(str(path))
    assert matches == []


def test_no_stale_old_package_name_references_remain() -> None:
    old_name = "cube_" + "shockfinder"
    matches: list[str] = []
    for path in _iter_repo_text_files():
        text = path.read_text()
        if old_name in text:
            matches.append(str(path))
    assert matches == []


def test_shock_finder_config_has_no_chunk_size() -> None:
    field_names = {field.name for field in fields(ShockFinderConfig)}
    assert "chunk_size" not in field_names


def test_finder_dispatches_to_chunked_workflow_when_needed() -> None:
    finder_text = Path("shockit/finder.py").read_text()
    assert "cube.is_chunked" in finder_text
    assert "output=..." in finder_text
    assert "run_chunked_shock_finder" in finder_text
    assert "_run_chunked_pass" not in finder_text
    assert "_use_chunking" not in finder_text


def test_core_physics_modules_do_not_import_chunking() -> None:
    for path in (
        Path("shockit/derived.py"),
        Path("shockit/gradients.py"),
        Path("shockit/sampling.py"),
        Path("shockit/masks.py"),
        Path("shockit/mach.py"),
    ):
        text = path.read_text()
        assert "chunking" not in text


def test_readme_links_review_map() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/review_map.md](docs/review_map.md)" in readme


def test_readme_documents_explicit_chunked_workflow() -> None:
    readme = Path("README.md").read_text()
    assert "The same `FluidCube` and `ShockFinder` API is used in both cases." in readme
    assert "Chunking is inferred from field storage, not from array size." in readme
    assert "ShockFinder(config).find(cube)" in readme
    assert 'rho="chunked_snapshot/rho"' in readme
