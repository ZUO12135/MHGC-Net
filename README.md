# MHGC-Net

Core model implementation for **MHGC-Net: A Multi-View Heterogeneous Graph Collaborative Attention Network for Robust Audio Deepfake Detection**.

The model source is provided in `models/`. The complete training and evaluation code will be publicly released upon acceptance of the manuscript.

Default augmentation probabilities (Table 3): `lsb_prob=0.05`, `sfr_prob=0.05`.

Dependencies: PyTorch, NumPy, and fairseq. Model initialization requires external XLS-R and ACRNN pretrained assets, which are not included in this release.

Acknowledgements: [AASIST](https://github.com/clovaai/aasist), [Speech-Emotion-Recognition](https://github.com/Chien-Hung/Speech-Emotion-Recognition), [SSL Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing), [AMSDF](https://github.com/ItzJuny/AMSDF24), and [fairseq](https://github.com/facebookresearch/fairseq). Original license and copyright notices are retained.
