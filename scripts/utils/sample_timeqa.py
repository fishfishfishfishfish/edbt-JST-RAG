#!/usr/bin/env python3
"""Create a small, reproducible TimeQA subset for RQ2 smoke tests.

The RQ2 loader expects an ``annotated_<split>.json`` file whose top-level
value is a JSON list.  Sampling whole top-level records keeps each entity's
paragraphs, questions, answers, and temporal validity annotations together.

Examples:
    python scripts/utils/sample_timeqa.py
    python scripts/utils/sample_timeqa.py --sample-size 20 --seed 7
    python scripts/utils/sample_timeqa.py --dry-run --sample-size 5
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "TimeQA"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "TimeQA-rq2-sample"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample complete TimeQA records for an RQ2 smoke test."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing annotated split files (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the sampled split file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="dev",
        help="TimeQA split to sample; RQ2 uses dev by default (default: dev)",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=10,
        help="Number of top-level TimeQA records to retain (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to make the subset reproducible (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing sampled split file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and summarize records without writing the output file",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"TimeQA split file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON list in {input_path}, got {type(records).__name__}"
        )
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every TimeQA record in {input_path} must be a JSON object")
    return records


def sample_records(
    records: list[dict[str, Any]], sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[int]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero")
    if sample_size > len(records):
        raise ValueError(
            f"--sample-size ({sample_size}) exceeds the number of available "
            f"records ({len(records)})"
        )

    selected_indices = sorted(random.Random(seed).sample(range(len(records)), sample_size))
    return [records[index] for index in selected_indices], selected_indices


def write_records(
    output_path: Path,
    records: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def count_nested_items(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Return approximate question and passage counts for supported TimeQA schemas."""
    question_count = 0
    passage_count = 0

    for record in records:
        if "questions" in record:
            questions = record.get("questions") or []
            question_count += len(questions) if isinstance(questions, list) else 0
        elif record.get("question"):
            question_count += 1

        if "paras" in record:
            paragraphs = record.get("paras") or []
            if isinstance(paragraphs, list):
                passage_count += sum(bool(str(paragraph).strip()) for paragraph in paragraphs)
        elif str(record.get("context") or "").strip():
            passage_count += 1

    return question_count, passage_count


def main() -> int:
    args = parse_args()
    input_path = args.source_dir.resolve() / f"annotated_{args.split}.json"
    output_path = args.output_dir.resolve() / f"annotated_{args.split}.json"

    if input_path == output_path:
        raise ValueError("Source and output paths must be different")

    records = load_records(input_path)
    sampled_records, selected_indices = sample_records(records, args.sample_size, args.seed)
    question_count, passage_count = count_nested_items(sampled_records)

    if not args.dry_run:
        write_records(output_path, sampled_records, overwrite=args.overwrite)

    print(f"Source: {input_path}")
    print(f"Available records: {len(records)}")
    print(f"Selected records: {len(sampled_records)}")
    print(f"Selected source indices: {selected_indices}")
    print(f"Questions in subset: {question_count}")
    print(f"Non-empty passages in subset: {passage_count}")
    print(f"Random seed: {args.seed}")
    if args.dry_run:
        print(f"Dry run: no file written (target would be {output_path})")
    else:
        print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
