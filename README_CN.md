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

物理信息神经网络（PINNs）通过在配点上最小化残差来求解偏微分方程（PDEs），但对所有残差一视同仁，常使训练在解含有陡峭层、移动锋面、逐渐变陡的界面或长时间输运时变慢或精度不足。本文提出课程引导高斯混合 PINN（CGMPINN）：周期性地对 PDE 残差的取值拟合一维高斯混合模型（GMM），并把由此得到的残差空间聚类映射回配点，作为损失权重。每个混合分量有一个难度分数（即该分量的条件均方残差）和一个精度因子；一个课程参数将关注点从容易、集中的聚类逐步转向困难的聚类。由于权重在分量内共享并归一化为单位均值，它们具有显式的上下界。我们证明了加权与标准 PDE 损失的一致可比性及近最优性保证、所得时变目标梯度范数的 $O(1/K)$ 界，以及带显式加权偏差项的 Rademacher 复杂度泛化界。在八个基准问题上（从椭圆型、抛物型问题到界面逐渐变陡的 Allen–Cahn 方程和长时间平流问题），CGMPINN 在六种对比方法中取得五次独立运行平均相对 $L_2$ 误差最低的结果，在相近训练时间下相对标准 PINN 降低 55%–98%。消融、超参数敏感性和重采样实验进一步考察了各组成部分的作用。

## 方法亮点

- 使用高斯混合模型对 PDE 残差取值建模，并把残差空间的聚类映射回配点作为损失权重。
- 通过平滑课程参数实现从简单到复杂的动态训练。
- 使用基于精度的调制机制，在训练早期侧重集中的残差聚类。
- 可与自适应损失平衡策略结合。
- 在八个典型 PDE 基准问题上进行实验，覆盖椭圆型、抛物型、双曲型、对流主导型和非线性反应扩散型问题，其中包括 Allen–Cahn 方程和长时间平流方程。

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

每个 `*_code` 文件夹包含 CGMPINN 及对比方法的训练和绘图脚本；每个 `*_parameter` 文件夹包含已保存的模型参数；`results` 文件夹包含多次独立运行的汇总表（见[结果汇总](#结果汇总)）。

## 基准问题

| 问题 | 类型 | 目录 |
| --- | --- | --- |
| 1D Poisson | 椭圆型 | `1D_poisson/` |
| 2D Poisson | 椭圆型 | `2D_poisson/` |
| 1D Heat | 抛物型 | `1D_heat/` |
| 1D Wave | 双曲型 | `1D_wave/` |
| 1D Advection-Diffusion | 对流主导型 | `1D_advection_diffusion/` |
| 1D Fisher-KPP | 非线性反应扩散型 | `1D_fisher_kpp/` |
| 1D Allen-Cahn | 界面逐渐变陡的非线性反应扩散型 | `1D_allen_cahn/` |
| 1D 长时间平流 | 长时间区间上的纯平流 | `1D_advection_long/` |

## 包含的方法

本仓库包含以下方法的实现：

- `CGMPINN`：本文提出的课程引导高斯混合 PINN。
- `PINN`：标准物理信息神经网络。
- `gPINN`：梯度增强 PINN。
- `lbPINN`：损失平衡 PINN。
- `LNN-PINN`：局部自适应神经网络 PINN。
- `STAR-PINN`：自训练自适应残差 PINN。
- `ablation`：CGMPINN 的消融实验版本（GMMPINN、CLPINN，以及去掉精度因子的 CGMPINN）。

## 环境安装

创建 Python 环境后安装依赖：

```bash
pip install numpy torch matplotlib scikit-learn scipy
```

如果使用 CUDA，请根据你的 GPU 和 CUDA 版本从 PyTorch 官网安装对应版本。注意：脚本默认在 CPU 上运行（所有张量都创建在 CPU 上）。

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

## 输出

- 训练脚本会把训练好的 Adam→L-BFGS（Tanh）模型保存到对应的 `*_parameter` 文件夹，并覆盖其中的同名文件。
- 绘图脚本在权重文件缺失或与模型结构不匹配时会直接报错。
- 所有图片以 PDF（600 dpi）格式保存到 `<算例>/<算例>_picture/`，例如 `1D_heat/1D_heat_picture/1D_heat_plot_comparison_group_loss.pdf`。该文件夹会自动创建，且不纳入 git 版本管理。

## 结果汇总

`results/` 文件夹包含多次独立运行的汇总表（均值与样本标准差）：

| 文件 | 内容 |
| --- | --- |
| `main_comparison.csv` | 八个基准问题上的全部方法；`cpu_time_s` 为 `*_parameter` 中所提供参数对应运行的 CPU 训练时间 |
| `ablation.csv` | 八个基准问题上的 GMMPINN、CLPINN、去掉精度因子的 CGMPINN 与 CGMPINN |
| `sensitivity.csv` | 1D Poisson 与 1D Heat 上 CGMPINN 对 `G`、`k_upd`、`beta`、`c_sat` 的单因素敏感性 |
| `resampling.csv` | 1D Poisson、1D Advection-Diffusion、1D Fisher-KPP 上固定配点与重采样内部配点的对比 |

列含义：`rel_e2`（相对 $L_2$ 误差）、`e2`（$L_2$ 误差）、`e_inf`（最大绝对误差）、`e_loss`（各方法自身训练目标的最终值；不同损失加权的方法之间不可直接比较）。

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
