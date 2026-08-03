# Failure-recovery dataset assessment

- Integrity: `PASS`
- Fixed-action oracle-label replay: `FAIL`
- Failure states: `37`
- Branchable states: `24` (64.9%)
- Branches / preference pairs: `278` / `167`
- Training admission: `FAIL`

Training is admitted only when integrity, fixed-action label replay, and persistent
state replay pass, the requested
minimum number of branchable states exists, and all three state-grouped splits are populated.
