# protein-stability-uncertainty

Sequence -> protein **melting point** regression with calibrated
uncertainty, evaluated on a homology-aware held-out split — the protein
analog of scaffold-split evaluation. Thermal stability is a food/protein
engineering trait (processing robustness, formulation shelf-life), and
the deliverable shape — a prediction *plus an honest interval and an
applicability domain* — is the part that transfers to decision support.

**Status: working demonstration.** Snakemake DAG: fetch -> parse ->
ESM-2 embed -> ridge -> split-conformal + applicability-domain eval.

## Design

- **Dataset**: FLIP meltome `mixed` split (Jarzab et al., Nature Methods
  2020 thermal proteome atlas, via the FLIP mirror): 27,951 proteins,
  target = melting point °C. The split is precomputed from MMSeqs2 20%
  sequence-identity clusters — held-out clusters share <=20% identity
  with anything in training. No random-split optimism.
- **Encoder under test**: ESM-2 (`esm2_t6_8M`, mean-pooled last hidden
  state, sequences truncated at 1024 aa — count logged). Baseline:
  amino-acid composition + log length. If ESM-2 can't beat composition
  features, the LM earns nothing here.
- **Model**: ridge regression — deliberately boring; the point is the
  evaluation wrapper, not the regressor.
- **Uncertainty**: split-conformal. Ridge is fit on train-minus-
  validation; absolute residuals calibrated on FLIP's `validation`
  subset (still train-side); 90% target coverage measured on held-out
  clusters. Calibration never touches test.
- **Applicability domain**: Euclidean distance in embedding space to the
  training centroid, quartile-binned on test — does coverage degrade
  where the model extrapolates?

## Result

(committed in `results/summary.json` — populated by the DAG run)

## Caveats

- Melting points are assay measurements with real noise; the conformal
  intervals bound *model* error, not experimental reproducibility.
- Mixed-species corpus — the test split is homology-separated, not
  species-stratified; cross-species distribution shift is present and
  not isolated as a factor.
- AD quartiles are descriptive (measured coverage per bucket), not a
  calibrated guarantee conditioned on the domain flag.
- Ridge on mean-pooled embeddings loses positional information;
  per-residue or attention-pooled features are the known upgrade path.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> parse -> embed -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`PROTSTAB_CONFIG` selects an alternate config; `encoder.kind:
composition` reproduces the no-LM baseline as a standalone run.
Embeddings cache to `data/processed/*_esm2.npy` keyed by encoder config +
exact sequence list — a changed model or context silently invalidates
the cache rather than reusing it.

## Data

FLIP `splits/meltome/splits.zip` (CC BY 4.0; meltome atlas per Jarzab et
al., Nature Methods 2020). Downloaded zip and cached embeddings are
gitignored; the compact summary + provenance are committed.
