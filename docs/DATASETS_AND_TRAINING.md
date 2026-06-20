# Real datasets and model training

HQCA can train on CSV exports prepared from BindingDB, PDBbind, or a custom
binding dataset. Large upstream datasets are not committed to this repository;
place them under a local data directory such as `data/raw/`.

## Supported CSV columns

At minimum, provide a SMILES column and one binding label column.

Recognized SMILES columns:

- `smiles`
- `SMILES`
- `Ligand SMILES`
- `ligand_smiles`
- `canonical_smiles`

Recognized optional sequence columns:

- `fasta`
- `FASTA`
- `protein_sequence`
- `target_sequence`
- `sequence`

Recognized binding labels:

- Direct energy: `binding_energy_kcal_mol`, `delta_g_kcal_mol`, `DeltaG`, `dG`
- pKd-style values: `pKd`, `pKi`, `affinity_pKd`, `-logKd/Ki`
- nM affinity values: `affinity_nM`, `Kd (nM)`, `Ki (nM)`, `IC50 (nM)`, `EC50 (nM)`

For nM affinity values, the training pipeline converts affinity to binding free
energy using:

```text
DeltaG = R * T * ln(Kd_M)
R = 0.0019872041 kcal/(mol*K)
T = 298.15 K
Kd_M = affinity_nM * 1e-9
```

## Binding energy to score formula

Prediction output uses a documented linear score:

```text
score = 100 * (E_max - E) / (E_max - E_min)
E_min = -15.0 kcal/mol
E_max = -0.1 kcal/mol
```

The score is clipped to `[0, 100]`; stronger, more negative binding energies
produce higher scores.

## Train a model

```bash
python training.py \
  --dataset data/raw/bindingdb_subset.csv \
  --source bindingdb \
  --artifact-dir models \
  --model-type random_forest \
  --n-estimators 200
```

Outputs:

- `models/hqca_model.joblib` — RandomForest baseline artifact
- `models/vqc_params.json` — saved VQC surrogate parameter vector
- `models/metrics.json` — train/test MAE and score MAE
- `models/training_records.csv` — records that passed RDKit featurization

The baseline model uses RDKit descriptors normalized with the same descriptor
pipeline used by the API. `--model-type random_forest` is the default. An
optional XGBoost baseline is available with:

```bash
python -m pip install -e ".[xgboost]"
python training.py --dataset data/raw/bindingdb_subset.csv --model-type xgboost
```

The VQC parameters are currently a supervised ridge surrogate over the seven
normalized descriptor inputs; they are persisted so the artifact layout can be
replaced by PennyLane/Qiskit optimized VQC parameters without changing
downstream consumers.

## Use the trained artifact in the API

```bash
HQCA_MODEL_ARTIFACT=models/hqca_model.joblib uvicorn api:app --reload
```

If `HQCA_MODEL_ARTIFACT` points to an existing artifact, `/predict` uses the
trained RandomForest baseline for `binding_energy_kcal_mol`. If the artifact is
missing, the API falls back to the existing VQE/classical simulator in `data.py`.

## CI fixture

`tests/fixtures/bindingdb_sample.csv` is a tiny smoke-test fixture that exercises
the BindingDB/PDBbind ingestion, featurization, MAE calculation, model saving,
and VQC-parameter saving paths. It is not a scientific benchmark. Real MAE
claims should be made only with an independently curated BindingDB/PDBbind split.
