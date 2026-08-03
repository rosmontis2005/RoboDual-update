# Aborted diagnostic collection

This run was intentionally stopped after four committed states. The four states came
from only two failed subtasks because `states_per_failed_subtask=2`; all 24 branches
failed. Continuing would have repeated v1's correlated-source problem. No files were
deleted and this partial directory is excluded from analysis and training.

The replacement collection uses one state per failed subtask and writes to
`collected_failure_recovery_v2_independent`.
