# From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models

<p align="center">
  <a href="https://arxiv.org/abs/2605.19263"><img src="https://img.shields.io/badge/arXiv-2605.19263-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2605.19263"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Papers-yellow" alt="Hugging Face"></a>
  <a href="https://github.com/Mathematics-Yang/CGMPINN"><img src="https://img.shields.io/badge/Code-CGMPINN-blue" alt="Code"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x%2F2.x-ee4c2c" alt="PyTorch"></a>
</p>

Official implementation of **Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network (CGMPINN)**.

[English](README.md) | [中文](README_CN.md)

CGMPINN uses Gaussian mixture modeling to estimate the spatially varying difficulty of PDE residuals and applies a smooth curriculum schedule that progressively shifts optimization from easier regions to harder regions. The method can also be combined with self-adaptive loss balancing.

> **Paper:** [arXiv:2605.19263](https://arxiv.org/abs/2605.19263)  
> **Hugging Face:** [https://huggingface.co/papers/2605.19263](https://huggingface.co/papers/2605.19263)  
> **Code:** https://github.com/Mathematics-Yang/CGMPINN

## Abstract

Physics-informed neural networks (PINNs) solve partial differential equations (PDEs) by minimizing residuals at collocation points, but treating all residuals uniformly often leads to slow or inaccurate training when the solution contains steep layers, moving fronts, sharpening interfaces, or long-time transport. We propose the curriculum-guided Gaussian mixture PINN (CGMPINN), which periodically fits a one-dimensional Gaussian mixture model (GMM) to the values of the PDE residuals and maps the resulting residual-space clusters back to the collocation points as loss weights. Each mixture component receives a difficulty score, given by its conditional mean squared residual, and a precision factor; a single curriculum parameter shifts the emphasis from easy, concentrated clusters to hard ones. Because the weights are shared within components and normalized to unit mean, they admit explicit two-sided bounds. We establish the uniform comparability of the weighted and standard PDE losses together with a near-optimality guarantee, an $O(1/K)$ bound on the gradient norm of the resulting time-varying objective, and a Rademacher-complexity generalization bound with an explicit weighting-induced bias. On eight benchmarks, ranging from elliptic and parabolic problems to an Allen–Cahn equation with sharpening interfaces and a long-time advection problem, CGMPINN attains the lowest mean relative $L_2$ error over five independent runs among six compared methods, reducing it by 55%–98% relative to the standard PINN at a comparable training time. Ablation, hyperparameter-sensitivity, and resampling studies further examine the role of each component.

## Highlights

- Gaussian mixture modeling of the PDE residual values; the residual-space clusters are mapped back to the collocation points as loss weights.
- Dynamic easy-to-hard curriculum training controlled by a smooth curriculum parameter.
- Precision-based modulation that emphasizes concentrated residual clusters early in training.
- Compatibility with adaptive loss balancing.
- Experiments on eight benchmark PDEs covering elliptic, parabolic, hyperbolic, advection-dominated, and nonlinear reaction-diffusion problems, including an Allen–Cahn equation and a long-horizon advection equation.

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
|-- 1D_fisher_kpp/
|   |-- 1D_fisher_kpp_code/
|   `-- 1D_fisher_kpp_parameter/
|-- 1D_allen_cahn/
|   |-- 1D_allen_cahn_code/
|   `-- 1D_allen_cahn_parameter/
|-- 1D_advection_long/
|   |-- 1D_advection_long_code/
|   `-- 1D_advection_long_parameter/
`-- results/
```

Each `*_code` directory contains training and plotting scripts for CGMPINN and baseline methods. Each `*_parameter` directory contains saved model parameters. The `results` directory contains summary tables over repeated runs (see [Results](#results)).

## Benchmarks

| Benchmark | Type | Directory |
| --- | --- | --- |
| 1D Poisson | Elliptic | `1D_poisson/` |
| 2D Poisson | Elliptic | `2D_poisson/` |
| 1D Heat | Parabolic | `1D_heat/` |
| 1D Wave | Hyperbolic | `1D_wave/` |
| 1D Advection-Diffusion | Advection-dominated | `1D_advection_diffusion/` |
| 1D Fisher-KPP | Nonlinear reaction-diffusion | `1D_fisher_kpp/` |
| 1D Allen-Cahn | Nonlinear reaction-diffusion with sharpening interfaces | `1D_allen_cahn/` |
| 1D Long-Horizon Advection | Pure advection over a long time horizon | `1D_advection_long/` |

## Methods Included

The repository contains implementations of:

- `CGMPINN`: the proposed curriculum-guided Gaussian mixture PINN.
- `PINN`: standard physics-informed neural network.
- `gPINN`: gradient-enhanced PINN.
- `lbPINN`: loss-balanced PINN.
- `LNN-PINN`: locally adaptive neural network PINN.
- `STAR-PINN`: self-training adaptive residual PINN.
- `ablation`: ablation variants of CGMPINN (GMMPINN, CLPINN, and CGMPINN without the precision factor).

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy torch matplotlib scikit-learn scipy
```

If you use CUDA, install the PyTorch build that matches your GPU and CUDA version from the official PyTorch website. Note that the scripts run on the CPU by default (all tensors are created on the CPU).

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

## Outputs

- Training scripts save the trained Adam→L-BFGS (Tanh) model to the corresponding `*_parameter` directory, overwriting the provided file with the same name.
- The plotting scripts raise an error if a weight file is missing or does not match the model architecture.
- All figures are saved as PDF (600 dpi) to `<benchmark>/<benchmark>_picture/`, e.g. `1D_heat/1D_heat_picture/1D_heat_plot_comparison_group_loss.pdf`. The folder is created automatically and is not tracked by git.

## Results

The `results/` directory contains summary tables (mean and sample standard deviation over independent runs) for:

| File | Content |
| --- | --- |
| `main_comparison.csv` | All methods on the eight benchmarks; `cpu_time_s` is the CPU training time of the run whose parameters are provided in `*_parameter` |
| `ablation.csv` | GMMPINN, CLPINN, CGMPINN without the precision factor, and CGMPINN on the eight benchmarks |
| `sensitivity.csv` | One-at-a-time sensitivity of CGMPINN to `G`, `k_upd`, `beta`, and `c_sat` on 1D Poisson and 1D Heat |
| `resampling.csv` | Fixed versus resampled interior collocation points on 1D Poisson, 1D Advection-Diffusion, and 1D Fisher-KPP |

Columns: `rel_e2` (relative $L_2$ error), `e2` ($L_2$ error), `e_inf` (maximum absolute error), and `e_loss` (final value of each method's own training objective; not comparable across methods with different loss weightings).

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
