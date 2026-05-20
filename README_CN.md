# 从简单到复杂：基于高斯混合模型的课程引导物理信息神经网络

[English](README.md) | [中文](README_CN.md)

<p align="center">
  <a href="https://arxiv.org/abs/2605.19263"><img src="https://img.shields.io/badge/arXiv-2605.19263-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2605.19263"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Papers-yellow" alt="Hugging Face"></a>
  <a href="https://github.com/Mathematics-Yang/CGMPINN"><img src="https://img.shields.io/badge/Code-CGMPINN-blue" alt="Code"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x%2F2.x-ee4c2c" alt="PyTorch"></a>
</p>

本仓库为 **Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network (CGMPINN)** 的官方实现。

CGMPINN 使用高斯混合模型对 PDE 残差分布进行建模，从而估计不同空间区域的学习难度，并通过平滑的课程学习策略，将训练重点从较容易区域逐步转移到较困难区域。该方法也可以与自适应损失平衡方法结合使用。

> **论文：** [arXiv:2605.19263](https://arxiv.org/abs/2605.19263)  
> **Hugging Face：** [https://huggingface.co/papers/2605.19263](https://huggingface.co/papers/2605.19263)  
> **代码：** https://github.com/Mathematics-Yang/CGMPINN

## 摘要

物理信息神经网络（PINNs）为求解偏微分方程（PDEs）提供了一种无网格框架，但其训练过程通常会受到梯度病态、谱偏差以及收敛困难等问题的影响，尤其是在处理强非线性、尖锐梯度或多尺度特征问题时更为明显。本文提出了课程引导高斯混合物理信息神经网络（Curriculum-Guided Gaussian Mixture Physics-Informed Neural Network, CGMPINN），将高斯混合建模与动态课程学习相结合。具体而言，CGMPINN 周期性地对 PDE 残差分布拟合高斯混合模型，以量化空间变化的学习难度；随后通过平滑的课程调度策略，将训练关注点从容易区域逐步转移到困难区域，同时利用基于精度的方差调制机制在训练早期抑制不可靠聚类的影响。该双重课程机制由共享课程参数控制，并可与自适应损失平衡结合。本文进一步给出了理论保证，包括诱导时变损失的梯度范数次线性收敛、课程加权 PDE 损失与标准 PDE 损失之间的一致等价性，以及具有显式加权偏差刻画的泛化界。六个涵盖椭圆型、抛物型、双曲型、对流主导型以及非线性反应扩散型 PDE 的基准实验表明，CGMPINN 在所有对比方法中稳定取得最低的相对 $L_2$ 误差和最大绝对误差，并在相近计算成本下，相比标准 PINN 最多降低 97.8% 的相对 $L_2$ 误差。

## 方法亮点

- 使用高斯混合模型对 PDE 残差进行建模，估计区域级学习难度。
- 通过平滑课程参数实现从简单到复杂的动态训练。
- 使用基于精度的方差调制机制，降低训练早期不可靠聚类的影响。
- 可与自适应损失平衡策略结合。
- 在六个典型 PDE 基准问题上进行实验，覆盖椭圆型、抛物型、双曲型、对流主导型和非线性反应扩散型问题。

## 仓库结构

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

每个 `*_code` 文件夹包含 CGMPINN 及对比方法的训练和绘图脚本；每个 `*_parameter` 文件夹包含已保存的模型参数。

## 基准问题

| 问题 | 类型 | 目录 |
| --- | --- | --- |
| 1D Poisson | 椭圆型 | `1D_poisson/` |
| 2D Poisson | 椭圆型 | `2D_poisson/` |
| 1D Heat | 抛物型 | `1D_heat/` |
| 1D Wave | 双曲型 | `1D_wave/` |
| 1D Advection-Diffusion | 对流主导型 | `1D_advection_diffusion/` |
| 1D Fisher-KPP | 非线性反应扩散型 | `1D_fisher_kpp/` |

## 包含的方法

本仓库包含以下方法的实现：

- `CGMPINN`：本文提出的课程引导高斯混合 PINN。
- `PINN`：标准物理信息神经网络。
- `gPINN`：梯度增强 PINN。
- `lbPINN`：损失平衡 PINN。
- `LNN-PINN`：局部自适应神经网络 PINN。
- `STAR-PINN`：自训练自适应残差 PINN。
- `ablation`：CGMPINN 的消融实验版本。

## 环境安装

创建 Python 环境后安装依赖：

```bash
pip install numpy torch matplotlib scikit-learn scipy
```

如果使用 CUDA，请根据你的 GPU 和 CUDA 版本从 PyTorch 官网安装对应版本。

## 快速开始

以 1D heat 方程为例运行 CGMPINN：

```bash
cd 1D_heat/1D_heat_code
python 1D_heat_CGMPINN.py
```

运行对应的标准 PINN：

```bash
python 1D_heat_PINN.py
```

生成图像：

```bash
python 1D_heat_plot.py
```

其他基准问题采用相同的命名规则。例如：

```bash
cd 1D_poisson/1D_poisson_code
python 1D_poisson_CGMPINN.py
python 1D_poisson_plot.py
```

## 预训练参数

已保存的模型参数位于对应的 `*_parameter` 文件夹中。例如：

```text
1D_heat/1D_heat_parameter/
|-- 1D_heat_tanh_adam2lbfgs_cgmpinn.pth
|-- 1D_heat_tanh_adam2lbfgs_pinn.pth
|-- 1D_heat_tanh_adam2lbfgs_gpinn.pth
`-- ...
```

这些参数可被绘图脚本读取，用于在不重新训练全部模型的情况下复现实验可视化结果。

## 引用

如果本仓库对你的研究有帮助，请引用我们的工作：

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

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

## 联系方式

如有问题，请联系Mathematics-Yang（JiananYang@stu.xjtu.edu.cn）。
