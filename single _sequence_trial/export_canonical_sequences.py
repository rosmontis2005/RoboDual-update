#!/usr/bin/env python3
"""Export the deterministic CALVIN get_sequences(100) catalog as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from calvin_agent.evaluation.multistep_sequences import get_sequences


def main() -> None:
    output = Path(__file__).resolve().parent / "canonical_100_sequences.json"
    sequences = []
    for index, (initial_state, tasks) in enumerate(get_sequences(100, num_workers=1)):
        sequences.append(
            {
                "sequence_index_zero_based": index,
                "sequence_number_human": index + 1,
                "initial_state": initial_state,
                "tasks": list(tasks),
            }
        )
    output.write_text(json.dumps(sequences, indent=2) + "\n")


if __name__ == "__main__":
    main()
