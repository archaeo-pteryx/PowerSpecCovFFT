# PowerSpecCovFFT
A Python code to compute the non-Gaussian (regular trispectrum and its shot noise) part of the analytic covariance matrix of the redshift-space galaxy power spectrum multipoles, using an FFTLog-based method proposed in [Kobayashi 2023](https://arxiv.org/abs/2308.08593). The trispectrum is based on standard perturbation theory as described in [Wadekar & Scoccimarro 2020](https://arxiv.org/abs/1910.02914), but with a slightly different galaxy bias expansion. 

The code includes the non-Gaussian covariance of the power spectrum monopole, quadrupole, hexadecapole, and their cross-covariance up to kmax ~ 0.4 h/Mpc.   

### Requirements

The following packages need to be installed.

- numpy
- scipy
- sympy

### Basic Usage

You can find an example Jupyter notebook [here](example/cov_non-gauss.ipynb). 

### Authors

- Yosuke Kobayashi (yosukekobayashi@arizona.edu)

### Citations

- Yosuke Kobayashi, Fast computation of non-Gaussian covariance of redshift-space galaxy power spectrum multipoles ([arXiv](https://arxiv.org/abs/2308.08593)) 

