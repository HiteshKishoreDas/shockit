"""Compatibility wrapper for chunk-aware center reduction."""

from __future__ import annotations

from pathlib import Path

from .chunking.centers import build_base_summary_from_output_store, finalize_chunked_shock_outputs
from .chunking.layout import load_chunk_layout
from .chunking.writer import NpzChunkedOutput
from .config import ShockFinderConfig


def reduce_chunked_shock_outputs(
    output_root: str | Path,
    config: ShockFinderConfig,
    *,
    progress: bool = True,
) -> dict[str, object]:
    """Compatibility wrapper around the chunking subpackage reducer."""

    output_path = Path(output_root)
    layout = load_chunk_layout(output_path / "full_shock_mask")
    output_store = NpzChunkedOutput(output_path, layout)
    summary = output_store.read_summary()
    if not summary:
        summary = build_base_summary_from_output_store(
            output_store,
            config,
            gamma=float("nan"),
        )
    return finalize_chunked_shock_outputs(
        output_store,
        config,
        summary,
        progress=progress,
    )
