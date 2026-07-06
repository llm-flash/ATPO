"""Convert records from prior validation steps into the format that 
compare_last_turn_completion.py expects. You need to go to logs/
(or wherever verl step records are being saved) and choose the 
directory of a previous run, then go to the validation folder and
pick one of the .jsonl files. That will be the input to this script.

The original
prompt and all but the last thinking/answer tokens from the LLM are
joined together into a new prompt which will be completed in 
compare_last_turn_completion.py. The ground truth is also extracted
from the original record to allow for checking answers.

Input format (per line of .jsonl):
{
    "input": "system\\n...\\nassistant\\n",
    "output": "<think>...</think>\\n\\n<answer> ... </answer>",
    "score": 1,
    "step": 10,
    "ground_truth": "[\"yes\"]",
}

Output format (per line of .jsonl):
{
    "prompt": "system\\n...\\nassistant\\n",
    "ground_truth": "yes",
    "score": 1,
    "step": 10,
}

Usage:
    python convert_prompt_format.py --input train_data.jsonl --output converted.jsonl
    python convert_prompt_format.py --input train_data.jsonl --output converted.jsonl --modify remove_thinking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_ground_truth(raw: str) -> str:
    """Extract the ground truth string from a JSON array string like '[\"yes\"]'."""
    """If the ground-truth is not wrapped in a JSON array, it will be returned as-is."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return str(parsed)
    except (json.JSONDecodeError, TypeError):
        return raw


def truncate_at_last_result(prompt: str) -> str:
    """Truncate the prompt at the last </result> marker, keeping everything up to and including it."""
    last_result = prompt.rfind("</result>")
    if last_result != -1:
        return prompt[: last_result + len("</result>")]
    return prompt


def truncate_before_thinking(prompt: str) -> str:
    """Truncate the prompt before the first <think> marker (remove all thinking)."""
    first_think = prompt.find("<think>")
    if first_think != -1:
        return prompt[: first_think + len("<think>")]
    return prompt


def modify_prompt(prompt: str, strategy: str) -> str:
    """Apply a named modification strategy to the prompt.

    Supported strategies:
      - truncate_result: truncate at the last </result> marker (default)
      - remove_thinking: truncate before the first <think> marker
      - truncate_result_or_thinking: try truncate_result first, fall back to remove_thinking
      - none: no modification
    """
    if strategy == "truncate_result":
        return truncate_at_last_result(prompt)
    if strategy == "remove_thinking":
        return truncate_before_thinking(prompt)
    if strategy == "truncate_result_or_thinking":
        truncated = truncate_at_last_result(prompt)
        if truncated == prompt:
            return truncate_before_thinking(prompt)
        return truncated
    if strategy == "none":
        return prompt
    raise ValueError(f"Unknown modify strategy: {strategy}")


def convert_record(record: dict, modify_strategy: str) -> dict:
    """Convert a single input record to the dummy_hotpotqa_prompt format."""
    input_text = record.get("input", "")
    output_text = record.get("output", "")
    combined_prompt = input_text + output_text
    combined_prompt = modify_prompt(combined_prompt, modify_strategy)

    ground_truth_raw = record.get("ground_truth", "[]")
    ground_truth = parse_ground_truth(ground_truth_raw)

    converted = {
        "prompt": combined_prompt,
        "ground_truth": ground_truth,
    }

    # Preserve unused fields
    for key in ("score", "step"):
        if key in record:
            converted[key] = record[key]

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert prompt format to dummy_hotpotqa_prompt style.")
    parser.add_argument("--input", required=True, help="Input .jsonl file path.")
    parser.add_argument("--output", required=True, help="Output .jsonl file path.")
    parser.add_argument(
        "--modify",
        choices=["truncate_result", "remove_thinking", "truncate_result_or_thinking", "none"],
        default="truncate_result_or_thinking",
        help="Prompt modification strategy (Should keep the default of truncate_result_or_thinking).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num}: invalid JSON: {e}", file=sys.stderr)
                continue
            converted = convert_record(record, args.modify)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} records from {input_path} -> {output_path} (modify={args.modify})")


if __name__ == "__main__":
    main()
