from dataclasses import fields
from pathlib import Path

from shockit import ShockFinderConfig


def test_no_root_prompt_md() -> None:
    assert not Path("prompt.md").exists()


def test_repository_contains_no_absolute_user_paths() -> None:
    matches: list[str] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if path.parts[0] in {".git", ".venv", "test_artifacts"}:
            continue
        if path == Path("tests/test_repo_hygiene.py"):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if "/Users/" in text:
            matches.append(str(path))
    assert matches == []


def test_no_stale_old_package_name_references_remain() -> None:
    old_name = "cube_" + "shockfinder"
    matches: list[str] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if path.parts[0] in {".git", ".venv", "test_artifacts"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if old_name in text:
            matches.append(str(path))
    assert matches == []


def test_shock_finder_config_has_no_chunk_size() -> None:
    field_names = {field.name for field in fields(ShockFinderConfig)}
    assert "chunk_size" not in field_names


def test_finder_stays_separate_from_chunking_modules() -> None:
    finder_text = Path("shockit/finder.py").read_text()
    assert "from .chunked" not in finder_text
    assert "from .chunking" not in finder_text
    assert "run_chunked_shock_finder" not in finder_text
    assert "_run_chunked_pass" not in finder_text
    assert "_use_chunking" not in finder_text


def test_readme_links_review_map() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/review_map.md](docs/review_map.md)" in readme


def test_readme_documents_explicit_chunked_workflow() -> None:
    readme = Path("README.md").read_text()
    assert "ShockFinder.find() never chunks automatically." in readme
    assert "chunked input store -> run_chunked_shock_finder() -> chunked output store + summary" in readme
