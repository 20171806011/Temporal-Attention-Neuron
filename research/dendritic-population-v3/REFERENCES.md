# Computational method references

- Neftci, Mostafa & Zenke (2019), *Surrogate Gradient Learning in Spiking Neural Networks*, https://arxiv.org/abs/1901.09948 . Background for using hard spikes in forward simulation with a substituted derivative for training. Our implementation and validation are custom; this citation does not validate its result.
- Gygax & Zenke (2024), *Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks*, https://arxiv.org/abs/2404.14964 . The authors explicitly distinguish surrogate updates from gradients of a globally defined surrogate loss. We verify a locally anchored graph derivative, not a true hard-network gradient or a global objective guarantee.
- Biological evidence and computational-analogy limitations remain in ../dendritic-population-v1/LITERATURE.md. V3 contributes no biological experiment or validation.
