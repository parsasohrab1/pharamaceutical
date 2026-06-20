# AGENTS.md

## Cursor Cloud specific instructions

### Overview
This repository is a single standalone Python script, `data.py` (the `README.md` is an
aspirational SRS document in Persian; the SaaS/React/FastAPI/DB architecture it describes is
**not implemented**). `data.py` is a synthetic-data generator for a quantum-classical
drug-discovery pipeline.

### Dependencies
- Required: `numpy`, `pandas`, `rdkit`.
- Optional: `pennylane` — enables the real VQE quantum-circuit path. Without it the script prints
  a notice and falls back to a classical linear model, but still runs to completion. It is
  installed by the update script so the quantum path is exercised.
- Installed system-wide via `pip install --break-system-packages ...` (Ubuntu marks the base
  Python as externally managed / PEP 668, so the flag is required). There is no virtualenv,
  lockfile, or dependency manifest in the repo.

### Running
- Run with `python3 data.py`. There is no build step, dev server, lint config, or test suite.
- The entry point hardcodes generation of **500** samples and writes `hqca_synthetic_data_500.csv`,
  `hqca_synthetic_data_500.json`, and `data_report.txt` **into the current working directory**.
  Run from a scratch dir (e.g. `/tmp`) if you don't want these artifacts in the repo root.
- Expected runtime is a few seconds.

### Non-obvious notes
- `SMILES Parse Error` lines on stderr are **normal**, not failures: the generator randomly
  mutates/crosses SMILES strings and uses RDKit to validate them, discarding invalid candidates.
- Output (descriptor stats, progress) is in Persian/Farsi; success ends with a `✅` line.
