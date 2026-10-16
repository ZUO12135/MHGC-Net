# MHGC-Net — Core Model Source

This release contains the MHGC-Net model architecture and its local building
blocks for code inspection. It is a model-source-only release. Trained MHGC-Net
weights, pretrained encoder assets, training/evaluation pipelines, datasets,
protocol keys, logs, scores, and reported-result reproduction are outside the
scope of this release.

The three model implementation files are copied byte-for-byte from the supplied
2026-08-27 source archive. No computation, constructor defaults, parameter names,
or forward behavior has been changed. See `SOURCE_MANIFEST.json` for provenance
and `MODEL_SOURCE_SHA256.txt` for source hashes.


## Release plan

The core model implementation is provided for review at this stage. The complete
training and evaluation code will be publicly released upon acceptance of the
manuscript. Model weights are not included in this core-source package.

## Included files

```text
models/
  __init__.py
  mhgc_net.py   # three view encoders, SVGE, MRGF, BCAM, HGRM, LSB/SFR, classifier
  AASIST.py     # graph attention, graph pooling, SincConv and residual blocks
  ACRNN.py      # prosodic encoder architecture
requirements-model.txt
LICENSE
THIRD_PARTY_NOTICES.md
SOURCE_MANIFEST.json
MODEL_SOURCE_SHA256.txt
VALIDATION.md
```

The LSB/SFR operations and the model's training-mode forward branch are retained
as part of the architecture. No training loop, optimizer, loss assembly,
checkpoint-selection procedure, or training launch script is included.

## Architecture and interface

The model uses acoustic, prosodic, and semantic views. Each is projected to
64-dimensional node features and passed through the single-view graph encoders
(SVGE), multi-relation graph fusion (MRGF), bidirectional cross-attention (BCAM),
and hierarchical graph readout (HGRM). A 384-to-64 projection feeds a two-class
classifier. The implementation also includes Latent Style Blending (LSB) and
Stochastic Feature Reorganization (SFR).

```python
from models.mhgc_net import MHGCNet

# Requires the external encoder assets described below.
model = MHGCNet()
model.eval()
# waveform: float32 tensor, shape (B, 64600), mono audio at 16 kHz
# prosodic_features: float32 tensor, shape (B, 3, 300, 40)
logits = model(waveform, prosodic_features, Freq_aug=False, stage="test")
# logits.shape == (B, 2)
```

The input feature contract in the supplied loader uses 40-filter log filter-bank
features at 16 kHz, their first and second deltas with window 2, and the first
300 frames. The three feature matrices are stacked as channels. Waveforms are
cropped or repeated to 64600 samples. The dataset loader and preprocessing
pipeline are not included; this description records their model input contract.

The output contains raw logits, not probabilities. Class 0 is spoof; class 1 is
bona fide. `return_feat=True` additionally returns the 64-dimensional feature
before the classifier. The documented inference call uses `stage="test"`.

## Dependencies and external assets

Direct model dependencies are PyTorch, NumPy, and fairseq. The environment
recorded in the supplied project used Python 3.9, PyTorch 2.8.0, torchaudio 2.8.0,
NumPy 2.0.2, and fairseq source snapshot
`a54021305d6b3c4c5959ac9395135f63202db8f1`. That complete environment has not been
revalidated for this source-only release. `requirements-model.txt` records the
direct pip dependencies; fairseq must be installed separately from its source
repository at the recorded revision.

Importing `models.mhgc_net` requires the dependencies above. Constructing
`MHGCNet()` additionally requires both of the following local files:

| Asset | Default location | Optional environment variable |
|---|---|---|
| XLS-R 300M pretrained asset | `pretrained_models/xlsr2_300m.pt` | `MHGC_XLSR_CHECKPOINT` |
| ACRNN pretrained asset | `pretrained_models/ser_acrnn.pth` | `MHGC_SER_CHECKPOINT` |

These binary assets are not included. The unchanged constructor will raise
`FileNotFoundError` when either asset is absent. Setting `active_views` does not
avoid these constructor requirements: all three encoders are constructed before
view masking. A trained MHGC-Net checkpoint is also needed to obtain meaningful
detector predictions; no such checkpoint is included in this release.

The legacy `sys.path` addition in `mhgc_net.py` is preserved for source fidelity.
Actual ACRNN and graph imports use the sibling `models/ACRNN.py` and
`models/AASIST.py`; the old `Speech-Emotion-Recognition-main` directory is not a
local source dependency of the model.

## Scope and verification

See `VALIDATION.md` for the checks performed and their limits. Model structure
inspection and component checks do not establish reproduction of paper results
or equivalence to a final manuscript. This release does not include a runnable
pretrained detector or a complete experimental reproduction pipeline.

The supplied license and upstream copyright notices are preserved. See
`LICENSE` and `THIRD_PARTY_NOTICES.md`.
