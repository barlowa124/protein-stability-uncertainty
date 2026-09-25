"""Sequence encoders.

esm2:        mean-pooled ESM-2 last hidden state (the claim under test —
             a protein LM should carry biophysical signal).
composition: the no-LM baseline — 20 amino-acid frequencies + log length.
             If ESM-2 can't beat this, the embedding earns nothing.
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def composition(seqs: pd.Series) -> np.ndarray:
    idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    X = np.zeros((len(seqs), len(AA_ALPHABET) + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = max(len(s), 1)
        counts = np.zeros(len(AA_ALPHABET))
        for c in s:
            j = idx.get(c)
            if j is not None:
                counts[j] += 1
        X[i, :-1] = counts / n
        X[i, -1] = np.log1p(n)
    return X


def esm2_embed(seqs: pd.Series, model_name: str = "facebook/esm2_t6_8M_UR50D",
               batch_size: int = 128, max_len: int = 1024) -> np.ndarray:
    """Mean-pooled last hidden state over residue positions.
    Sequences longer than `max_len` are truncated (counted by caller)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = [s[:max_len] for s in seqs]
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            enc = tok(seqs[i : i + batch_size], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_len).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out[i : i + batch_size] = (
                (hidden * mask).sum(1) / mask.sum(1)
            ).cpu().numpy()
    return out


def build_features(seqs: pd.Series, encoder_cfg: dict,
                   cache_stem: str = None) -> np.ndarray:
    kind = encoder_cfg.get("kind", "esm2")
    if kind == "composition":
        return composition(seqs)
    if kind != "esm2":
        raise ValueError(f"unknown encoder kind {kind!r}")
    if cache_stem:
        from pathlib import Path
        import hashlib
        import json

        npy = Path(f"{cache_stem}_esm2.npy")
        key_f = Path(f"{cache_stem}_esm2.key")
        # key covers encoder params + exact sequence list — a config change
        # must not silently reuse stale embeddings
        key = hashlib.sha256(
            (json.dumps(encoder_cfg, sort_keys=True)
             + "\n" + "\n".join(seqs)).encode()
        ).hexdigest()
        if npy.exists() and key_f.exists() and key_f.read_text() == key:
            return np.load(npy)
        X = esm2_embed(
            seqs,
            **{k: v for k, v in encoder_cfg.items() if k != "kind"})
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, X)
        key_f.write_text(key)
        return X
    return esm2_embed(
        seqs, **{k: v for k, v in encoder_cfg.items() if k != "kind"})
