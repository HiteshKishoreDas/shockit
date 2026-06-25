from pathlib import Path


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
    allowed = {"prompt.md"}
    old_name = "cube_" + "shockfinder"
    matches: list[str] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if path.parts[0] in {".git", ".venv", "test_artifacts"}:
            continue
        if path.name in allowed:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if old_name in text:
            matches.append(str(path))
    assert matches == []
