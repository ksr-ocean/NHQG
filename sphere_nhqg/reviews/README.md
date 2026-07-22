# Review Files

This directory contains the pedagogical review documents that accompany the
spherical solver code.

Review files are part of the implementation contract. When a Python module,
executable script, or nontrivial test is added or changed, its review should
be added or changed in the same patch. The review should be written for a
future reader checking the numerical method, not just the syntax.

Recommended review shape:

```text
# path/to/file.py

## Responsibility
## Numerical Assumptions
## Data Layout And Normalization
## Invariants And Tests
## Known Risks
```
