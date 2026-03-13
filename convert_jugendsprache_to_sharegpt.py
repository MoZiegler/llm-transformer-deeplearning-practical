#!/usr/bin/env python3
"""Convert jugendsprache_dataset_v1.json to ShareGPT train/eval JSON files.

Input format (list of objects):
  {
    "instruction": "...",
    "input": "...",
    "output": "..."
  }

Output format (ShareGPT list):
  {
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ],
    "source": "jugendsprache_dataset_v1",
    "quality": 1.0
  }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein Übersetzer zwischen Hochdeutsch und deutscher Jugendsprache. "
    "Übersetze den gegebenen Text in die jeweils andere Sprachvariante."
)


def build_human_prompt(instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()

    if instruction and user_input:
        return f"{instruction}\n\n{user_input}"
    if instruction:
        return instruction
    return user_input


def to_sharegpt_item(raw: dict[str, Any], system_prompt: str) -> dict[str, Any] | None:
    instruction = str(raw.get("instruction", "") or "")
    user_input = str(raw.get("input", "") or "")
    output = str(raw.get("output", "") or "")

    human_prompt = build_human_prompt(instruction, user_input)

    if not human_prompt or not output.strip():
        return None

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": output.strip()},
        ],
        "source": "jugendsprache_dataset_v1",
        "quality": 1.0,
    }


def convert_dataset(
    input_path: Path,
    train_path: Path,
    eval_path: Path,
    train_size: int,
    shuffle: bool,
    seed: int,
    system_prompt: str,
) -> tuple[int, int, int]:
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Expected input JSON to be a list of records.")

    converted: list[dict[str, Any]] = []
    skipped = 0

    for item in raw_data:
        if not isinstance(item, dict):
            skipped += 1
            continue

        sharegpt_item = to_sharegpt_item(item, system_prompt=system_prompt)
        if sharegpt_item is None:
            skipped += 1
            continue

        converted.append(sharegpt_item)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(converted)

    split_index = min(train_size, len(converted))
    train_data = converted[:split_index]
    eval_data = converted[split_index:]

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    return len(train_data), len(eval_data), skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert instruction/input/output JSON into ShareGPT train/eval JSON files."
    )
    parser.add_argument(
        "--input",
        default="jugendsprache_dataset_v1.json",
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--train-output",
        default="train.json",
        help="Path to output train JSON file.",
    )
    parser.add_argument(
        "--eval-output",
        default="eval.json",
        help="Path to output eval JSON file.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=450,
        help="Number of samples written to train output.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle examples before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used in each ShareGPT sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)

    train_count, eval_count, skipped = convert_dataset(
        input_path=input_path,
        train_path=train_path,
        eval_path=eval_path,
        train_size=args.train_size,
        shuffle=args.shuffle,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )

    print(f"Wrote {train_count} samples to {train_path}")
    print(f"Wrote {eval_count} samples to {eval_path}")
    print(f"Skipped {skipped} invalid/empty records")


if __name__ == "__main__":
    main()
