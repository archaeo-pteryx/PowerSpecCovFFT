# PowerSpecCovFFT

A Python code to compute the non-Gaussian (regular trispectrum and its shot noise) part of the analytic covariance matrix of the redshift-space galaxy power spectrum multipoles, using an FFTLog-based method proposed in [Kobayashi (2023)](https://arxiv.org/abs/2308.08593). The galaxy trispectrum is based on tree-level standard perturbation theory as described in [Wadekar & Scoccimarro (2020)](https://arxiv.org/abs/1910.02914), but with a slightly different galaxy bias expansion (see Appendix A of [Kobayashi (2023)](https://arxiv.org/abs/2308.08593)). 

The code computes the non-Gaussian covariance of the power spectrum monopole, quadrupole, hexadecapole, and their cross-covariance up to kmax ~ 0.4 h/Mpc.

### Installation

```bash
pip install powercovfft@git+https://github.com/archaeo-pteryx/PowerSpecCovFFT.git
```

### Basic Usage

You can find an example Jupyter notebook [here](example/cov_non-gauss.ipynb). 

### Authors

- Yosuke Kobayashi (yosukekobayashi@arizona.edu)

### Citations

- Yosuke Kobayashi, Fast computation of the non-Gaussian covariance of redshift-space galaxy power spectrum multipoles (2023, [arXiv](https://arxiv.org/abs/2308.08593), [Phys. Rev. D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.103512))
