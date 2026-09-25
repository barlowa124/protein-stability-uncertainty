# protein-stability-uncertainty

Sequence -> protein **melting point** regression with calibrated
uncertainty, evaluated on a homology-aware held-out split, the protein
analog of scaffold-split evaluation. Thermal stability is a food/protein
engineering trait (processing robustness, formulation shelf-life). The deliverable
is a prediction plus a calibrated interval and an applicability domain,
the part that transfers to decision support.

**Status: working demonstration.** Snakemake DAG: fetch -> parse ->
ESM-2 embed -> ridge -> split-conformal + applicability-domain eval.

## Design

- **Dataset**: FLIP meltome `mixed` split (Jarzab et al., Nature Methods
  2020 thermal proteome atlas, via the FLIP mirror): 27,951 proteins,
  target = melting point °C. The split is precomputed from MMSeqs2 20%
  sequence-identity clusters, so held-out clusters share <=20% identity
  with anything in training. No random-split optimism.
- **Encoder under test**: ESM-2 (`esm2_t6_8M`, mean-pooled last hidden
  state, sequences truncated at 1024 aa, count logged). Baseline:
  amino-acid composition + log length. If ESM-2 can't beat composition
  features, the LM earns nothing here.
- **Model**: ridge regression. What is under test is the evaluation
  wrapper, not the regressor.
- **Uncertainty**: split-conformal. Ridge is fit on train-minus-
  validation. Absolute residuals are calibrated on FLIP's `validation`
  subset (still train-side). 90% target coverage is measured on held-out
  clusters. Calibration never touches test.
- **Applicability domain**: Euclidean distance in embedding space to the
  training centroid, quartile-binned on test. Does coverage degrade
  where the model extrapolates?

## Result

(committed in `results/summary.json`, populated by the DAG run)

## Caveats

- Melting points are assay measurements with real noise. The conformal
  intervals bound *model* error, not experimental reproducibility.
- Mixed-species corpus. The test split is homology-separated, not
  species-stratified, so cross-species distribution shift is present and
  not isolated as a factor.
- AD quartiles are descriptive (measured coverage per bucket), not a
  calibrated guarantee conditioned on the domain flag.
- Ridge on mean-pooled embeddings loses positional information.
  Per-residue or attention-pooled features are the known upgrade path.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> parse -> embed -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`PROTSTAB_CONFIG` selects an alternate config. `encoder.kind:
composition` reproduces the no-LM baseline as a standalone run.
Embeddings cache to `data/processed/*_esm2.npy` keyed by encoder config
and exact sequence list. A changed model or context silently invalidates
the cache instead of reusing it.

## Data

FLIP `splits/meltome/splits.zip` (CC BY 4.0; meltome atlas per Jarzab et
al., Nature Methods 2020). Downloaded zip and cached embeddings are
gitignored. The compact summary and provenance are committed.
