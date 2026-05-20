# From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models

<p align="center">
  <a href="https://arxiv.org/abs/2605.19263"><img src="https://img.shields.io/badge/arXiv-2605.19263-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Mathematics-Yang/CGMPINN"><img src="https://img.shields.io/badge/Code-CGMPINN-blue" alt="Code"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x%2F2.x-ee4c2c" alt="PyTorch"></a>
</p>

Official implementation of **Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network (CGMPINN)**.

[English](README.md) | [中文](README_CN.md)

CGMPINN uses Gaussian mixture modeling to estimate the spatially varying difficulty of PDE residuals and applies a smooth curriculum schedule that progressively shifts optimization from easier regions to harder regions. The method can also be combined with self-adaptive loss balancing.

> **Paper:** [arXiv:2605.19263](https://arxiv.org/abs/2605.19263)  
> **Code:** https://github.com/Mathematics-Yang/CGMPINN

## Abstract

Physics-informed neural networks (PINNs) offer a mesh-free framework for solving partial differential equations (PDEs), yet training often suffers from gradient pathologies, spectral bias, and poor convergence, especially for problems with strong nonlinearity, sharp gradients, or multiscale features. We propose the Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network (CGMPINN), which integrates Gaussian mixture modeling with dynamic curriculum learning. Specifically, a GMM is periodically fitted to the PDE residual distribution to quantify spatially varying learning difficulty. A smooth curriculum schedule progressively shifts training focus from easy to harder regions, while precision-based variance modulation suppresses unreliable clusters during early optimization. This dual curriculum is governed by a shared curriculum parameter and can be combined with self-adaptive loss balancing. We further establish theoretical guarantees, including sublinear convergence of the gradient norm for the induced time-varying loss, uniform equivalence between the curriculum-weighted and standard PDE losses, and a generalization bound with an explicit weighting-induced bias characterization. Experiments on six benchmark PDEs spanning elliptic, parabolic, hyperbolic, advection-dominated, and nonlinear reaction-diffusion types show that CGMPINN consistently achieves the lowest relative $L_2$ and maximum absolute errors among all compared methods, reducing relative $L_2$ error by up to 97.8% over the standard PINN at comparable cost.

## Highlights

- Gaussian mixture modeling of PDE residuals to estimate region-wise learning difficulty.
- Dynamic easy-to-hard curriculum training controlled by a smooth curriculum parameter.
- Precision-based variance modulation to reduce the influence of unreliable clusters in early training.
- Compatibility with adaptive loss balancing.
- Experiments on six benchmark PDEs covering elliptic, parabolic, hyperbolic, advection-dominated, and nonlinear reaction-diffusion problems.

## Repository Structure

```text
CGMPINN/
|-- 1D_poisson/
|   |-- 1D_poisson_code/
|   `-- 1D_poisson_parameter/
|-- 2D_poisson/
|   |-- 2D_poisson_code/
|   `-- 2D_poisson_parameter/
|-- 1D_heat/
|   |-- 1D_heat_code/
|   `-- 1D_heat_parameter/
|-- 1D_wave/
|   |-- 1D_wave_code/
|   `-- 1D_wave_parameter/
|-- 1D_advection_diffusion/
|   |-- 1D_advection_diffusion_code/
|   `-- 1D_advection_diffusion_parameter/
`-- 1D_fisher_kpp/
    |-- 1D_fisher_kpp_code/
    `-- 1D_fisher_kpp_parameter/
```

Each `*_code` directory contains training and plotting scripts for CGMPINN and baseline methods. Each `*_parameter` directory contains saved model parameters.

## Benchmarks

| Benchmark | Type | Directory |
| --- | --- | --- |
| 1D Poisson | Elliptic | `1D_poisson/` |
| 2D Poisson | Elliptic | `2D_poisson/` |
| 1D Heat | Parabolic | `1D_heat/` |
| 1D Wave | Hyperbolic | `1D_wave/` |
| 1D Advection-Diffusion | Advection-dominated | `1D_advection_diffusion/` |
| 1D Fisher-KPP | Nonlinear reaction-diffusion | `1D_fisher_kpp/` |

## Methods Included

The repository contains implementations of:

- `CGMPINN`: the proposed curriculum-guided Gaussian mixture PINN.
- `PINN`: standard physics-informed neural network.
- `gPINN`: gradient-enhanced PINN.
- `lbPINN`: loss-balanced PINN.
- `LNN-PINN`: locally adaptive neural network PINN.
- `STAR-PINN`: self-training adaptive residual PINN.
- `ablation`: ablation variants of CGMPINN.

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy torch matplotlib scikit-learn scipy
```

If you use CUDA, install the PyTorch build that matches your GPU and CUDA version from the official PyTorch website.

## Quick Start

Run a CGMPINN experiment, for example on the 1D heat equation:

```bash
cd 1D_heat/1D_heat_code
python 1D_heat_CGMPINN.py
```

Run the corresponding baseline PINN:

```bash
python 1D_heat_PINN.py
```

Generate plots using the plotting script:

```bash
python 1D_heat_plot.py
```

The other benchmark folders follow the same naming convention. For example:

```bash
cd 1D_poisson/1D_poisson_code
python 1D_poisson_CGMPINN.py
python 1D_poisson_plot.py
```

## Pretrained Parameters

Saved model parameters are provided in the corresponding `*_parameter` directories. For example:

```text
1D_heat/1D_heat_parameter/
|-- 1D_heat_tanh_adam2lbfgs_cgmpinn.pth
|-- 1D_heat_tanh_adam2lbfgs_pinn.pth
|-- 1D_heat_tanh_adam2lbfgs_gpinn.pth
`-- ...
```

These files can be used by the plotting scripts to reproduce visual comparisons without retraining all models.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{yang2026simplecomplexcurriculumguidedphysicsinformed,
  title         = {From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models},
  author        = {Jianan Yang and Yiran Wang and Shuai Li and Fujun Cao and Xuefei Yan and Junmin Liu},
  year          = {2026},
  eprint        = {2605.19263},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.19263}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions, please contact Jianan Yang (JiananYang@stu.xjtu.edu.cn).
