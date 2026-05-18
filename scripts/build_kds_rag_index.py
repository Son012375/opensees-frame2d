"""Build the KDS-RAG index used by the /explain endpoint.

Usage:
    python scripts/build_kds_rag_index.py \\
        --source-dir data/kds \\
        --index-path data/kds_index.jsonl \\
        [--model voyage-4-large]

Behavior:
    * Loads .txt/.md/.json files under --source-dir.
    * Chunks them deterministically (see core.kds_rag.ingest).
    * Embeds each chunk via Voyage and writes a JSONL index.
    * Prints a one-line summary at the end.

Required environment:
    VOYAGE_API_KEY (or VOYAGEAI_API_KEY)

Exit codes:
    0  — index written successfully.
    2  — missing API key or unavailable SDK.
    3  — source directory missing or empty.
    1  — unexpected error.

The API key is read from the environment OR --api-key. It is never
logged. PDF files are skipped (Phase 4 work).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add mcp-server to path so core.* imports work whether the script is
# invoked directly or via a venv shim.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.kds_rag.ingest import build_kds_index  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Voyage-embedded KDS index for /explain.",
    )
    p.add_argument(
        "--source-dir",
        required=True,
        help="Directory of .txt/.md/.json KDS documents (recursive).",
    )
    p.add_argument(
        "--index-path",
        required=True,
        help="Destination JSONL index file.",
    )
    p.add_argument(
        "--model",
        default="voyage-4-large",
        help="Voyage embedding model (default: voyage-4-large).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Chunks per embed request (default: 32).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Voyage API key (else VOYAGE_API_KEY env var).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = args.api_key or os.environ.get("VOYAGE_API_KEY") \
        or os.environ.get("VOYAGEAI_API_KEY")
    if not api_key:
        print(
            "ERROR: VOYAGE_API_KEY (or VOYAGEAI_API_KEY) is required. "
            "Pass --api-key or set the environment variable.",
            file=sys.stderr,
        )
        return 2

    if not Path(args.source_dir).is_dir():
        print(
            f"ERROR: --source-dir not found or not a directory: "
            f"{args.source_dir}",
            file=sys.stderr,
        )
        return 3

    try:
        summary = build_kds_index(
            source_dir=args.source_dir,
            index_path=args.index_path,
            embedding_model=args.model,
            api_key=api_key,
            batch_size=args.batch_size,
        )
    except RuntimeError as exc:
        # Distinguish missing-SDK from no-docs without leaking API key.
        msg = str(exc)
        if "voyageai" in msg or "VOYAGE_API_KEY" in msg:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2
        if "No supported" in msg or "produced zero chunks" in msg:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 3
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: unexpected failure: {exc!r}", file=sys.stderr)
        return 1

    print(
        "OK: indexed "
        f"{summary['n_chunks']} chunks from {summary['n_docs']} docs "
        f"(model={summary['model']}, dim={summary['dim']}) "
        f"-> {summary['index_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
