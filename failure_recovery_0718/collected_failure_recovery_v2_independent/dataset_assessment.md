# Failure-recovery dataset assessment

- Integrity: `PASS`
- Fixed-action oracle-label replay: `FAIL`
- Failure states: `62`
- Branchable states: `31` (50.0%)
- Branches / preference pairs: `456` / `306`
- Training admission: `FAIL`

Training is admitted only when integrity, fixed-action label replay, and persistent
state replay pass, the requested
minimum number of branchable states exists, and all three state-grouped splits are populated.
