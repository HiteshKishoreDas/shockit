from __future__ import annotations

from pathlib import Path


def test_review_map_exists() -> None:
    assert Path("docs/review_map.md").is_file()


def test_test_suite_doc_exists() -> None:
    assert Path("docs/test_suite.md").is_file()


def test_testing_doc_exists() -> None:
    assert Path("docs/testing.md").is_file()


def test_chunked_center_reduction_doc_exists() -> None:
    assert Path("docs/chunked_center_reduction.md").is_file()


def test_readme_links_to_review_map() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/review_map.md](docs/review_map.md)" in readme


def test_readme_links_to_test_suite_doc() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/test_suite.md](docs/test_suite.md)" in readme


def test_readme_links_to_testing_docs() -> None:
    readme = Path("README.md").read_text()
    assert "[docs/testing.md](docs/testing.md)" in readme
    assert "[docs/chunked_center_reduction.md](docs/chunked_center_reduction.md)" in readme
