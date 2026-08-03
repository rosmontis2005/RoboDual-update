# Failure-recovery dataset assessment

- Integrity: `PASS`
- Fixed-action oracle-label replay: `PASS`
- Failure states: `20`
- Branchable states: `8` (40.0%)
- Branches / preference pairs: `180` / `86`
- Training admission: `PASS`

Training is admitted only when integrity, fixed-action label replay, and persistent
state replay pass, the requested
minimum number of branchable states exists, and all three state-grouped splits are populated.
