from __future__ import annotations

from pathlib import Path


def test_review_map_exists() -> None:
    assert Path("docs/review_map.md").is_file()


def test_readme_links_to_review_map() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/review_map.md](docs/review_map.md)" in readme
