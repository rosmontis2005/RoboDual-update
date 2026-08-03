"""List remaining CALVIN sequences by earliest task selected for recovery branching."""

import argparse
import importlib.util
from pathlib import Path
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve().parents[1] / "vla-scripts/evaluate_calvin_failure_recovery_scale_0718.py"
    sys.path.insert(0, script.parent.as_posix())
    spec = importlib.util.spec_from_file_location("recovery_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    allowlist = {item.strip() for item in args.tasks.split(",") if item.strip()}
    sequences = list(module.get_sequences(args.end))
    rows = []
    for sequence_i in range(args.start, args.end):
        tasks = list(sequences[sequence_i][1])
        positions = [index for index, task in enumerate(tasks) if task in allowlist]
        if positions:
            position = min(positions)
            rows.append((sequence_i, position, tasks[position], tasks))
    print(
        f"remaining={args.end - args.start} with_allowlist={len(rows)} "
        f"first_position_counts={dict(Counter(item[1] for item in rows))}"
    )
    for sequence_i, position, task, tasks in sorted(rows, key=lambda item: (item[1], item[0])):
        print(sequence_i, position, task, ",".join(tasks))


if __name__ == "__main__":
    main()
