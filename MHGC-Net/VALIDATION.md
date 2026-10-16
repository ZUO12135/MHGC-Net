# Model-source subset validation

Date: 2026-10-16 (Asia/Shanghai)

- Supplied source archive SHA256 matches its supplied checksum: PASS.
- All 23 Python files in the supplied source archive parse
  using the Python 3.9 grammar and compile under the audit interpreter: PASS.
- The four included Python files and the license match the supplied files
  byte-for-byte: PASS.
- Required local model imports (`.ACRNN`, `.AASIST`) are included: PASS.
- Acoustic encoder forward, random waveform `(1, 64600)`: `(1, 29, 64)`, PASS.
- ACRNN forward, random features `(1, 3, 300, 40)`: `(1, 150, 256)`, PASS.
- Graph fusion forward, synthetic 64-D view features with unequal node counts,
  batches 1 and 2: logits `(B, 2)` and features `(B, 64)`, finite outputs, PASS.
- LSB same-class permutation and finite shape-preserving output: PASS.
- SFR feature permutation: PASS.

Component-test environment: Python 3.12.4, PyTorch 2.7.0+cpu,
NumPy 1.26.4, CPU. This differs from the original project's recorded
environment.

fairseq was unavailable. An import-only fairseq placeholder was used for the
component checks; no fairseq API was called. External-asset encoder constructors
were bypassed only to construct and test the original graph fusion core on
synthetic view features. The acoustic encoder and ACRNN architecture were tested
separately with random weights.

Not tested: XLS-R loading/semantic encoder, pretrained ACRNN loading, full
raw-audio forward, strict loading of trained MHGC-Net checkpoints, backward
passes/training, dataset decoding, EER, or correspondence to final paper results.
The supplied historical `VALIDATION.md` was not copied because its checkpoint
and numerical-equivalence claims were not rerun in this audit.
