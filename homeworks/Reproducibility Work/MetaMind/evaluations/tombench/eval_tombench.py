import json
import os
import re
import argparse
import sys
import logging
from tqdm import tqdm

# Ensure project root is importable when running this script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import MetamindApplication

def _resolve_data_dir() -> str:
    """
    Handle both layouts:
    - evaluations/tombench/*.jsonl
    - evaluations/evaluations/tombench/*.jsonl (nested in evaluations.zip)
    """
    candidates = [
        os.path.join("evaluations", "tombench"),
        os.path.join("evaluations", "evaluations", "tombench"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            has_jsonl = any(name.endswith(".jsonl") for name in os.listdir(d))
            if has_jsonl:
                return d
    # Default to the first path for clearer error messages downstream
    return candidates[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaMind on ToMBench (Accuracy).")
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Optional substring filter for jsonl filename (e.g., 'Ambiguous Story Task'). Empty means all tasks.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Optional cap on number of examples per file. 0 means no cap.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Comma-separated example counts to print interim accuracy (e.g., '10,20,30,40,50').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, keep INFO logs from the MetaMind pipeline. Otherwise, reduce logging noise.",
    )
    args = parser.parse_args()

    if not args.verbose:
        # Reduce log noise from the pipeline during evaluation
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("MetamindApp").setLevel(logging.WARNING)

    checkpoints: set[int] = set()
    if args.checkpoints.strip():
        for part in args.checkpoints.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                checkpoints.add(int(part))
            except ValueError as e:
                raise ValueError(f"Invalid --checkpoints value '{part}'. Expected integers.") from e

    app = MetamindApplication()

    data_dir = _resolve_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    if args.task:
        files = [f for f in files if args.task.lower() in f.lower()]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found under '{data_dir}'. "
            "If you only have evaluations.zip, unzip it first."
        )

    total_correct = 0
    total_examples = 0

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if args.max_examples and args.max_examples > 0:
                lines = lines[: args.max_examples]
            for line in tqdm(lines):
                example = json.loads(line)
                
                story = example.get("STORY", "")
                question = example.get("QUESTION", "")
                option_a = example.get("OPTION-A", "")
                option_b = example.get("OPTION-B", "")
                option_c = example.get("OPTION-C", "")
                option_d = example.get("OPTION-D", "")
                # Some ToMBench JSONLs use a bilingual key like "答案\\nANSWER"
                answer = (
                    example.get("ANSWER")
                    or example.get("答案\nANSWER")
                    or example.get("答案")
                    or ""
                )

                options_str = f"A: {option_a}\nB: {option_b}\nC: {option_c}\nD: {option_d}"

                user_utterance = f"{story}\n\n{question}\n\nOptions:\n{options_str}\n\nPlease select the correct answer by outputting only the letter (A, B, C, or D)."

                result = app.process_user_input(user_utterance, conversation_context=[])
                response = result["final_response"]

                # Simple parsing: find the first A/B/C/D in the response
                match = re.search(r"\b[A-D]\b", response)
                predicted = match.group(0) if match else None

                if predicted == answer:
                    total_correct += 1

                total_examples += 1
                if checkpoints and (total_examples in checkpoints):
                    acc = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
                    print(
                        f"[Checkpoint] examples={total_examples} correct={total_correct} accuracy={acc:.2f}%",
                        flush=True,
                    )

    accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0
    if args.task:
        print(f"Task filter: {args.task}")
    print(f"Data dir: {data_dir}")
    print(f"Total examples: {total_examples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
