# Failure-recovery dataset assessment

- Integrity: `PASS`
- Fixed-action oracle-label replay: `PASS`
- Failure states: `16`
- Branchable states: `16` (100.0%)
- Branches / preference pairs: `34` / `18`
- Training admission: `PASS`

Training is admitted only when integrity, fixed-action label replay, and persistent
state replay pass, the requested
minimum number of branchable states exists, and all three state-grouped splits are populated.
