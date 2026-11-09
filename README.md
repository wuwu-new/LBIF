# Dual-Res-Forensics

局部监督驱动的有界增量融合人脸伪造检测（LBIF）。

此版本来自远端于 **2026年11月9日** 创建并校验的 **“代码第一版”** 快照，替换本仓库旧版训练与模型代码。它保存实际实验使用的源码、冻结配置、注册记录及核验材料，覆盖完整模型、五项匹配消融和三种子语义基座。

## 方法

框架冻结完整增强语义检测器，使用独立 CNN 从有符号高斯误差中学习局部表征，并结合掩膜派生的边界监督、块级分类与图内区域对比目标。局部证据通过零初始化的有界头生成 logit 修正；推理不需要真实掩膜。

## 代码入口

| 用途 | 路径 |
| --- | --- |
| 完整模型及局部学习目标 | [model_cuda.py](remote_runs/ffpp_patch_region_full_20261105/implementation/model_cuda.py) |
| 完整模型训练与原始评价 | [run_cuda.py](remote_runs/ffpp_patch_region_full_20261105/implementation/run_cuda.py) |
| 完整模型的预取目标域推理 | [run_cuda.py](remote_runs/ffpp_patch_region_prefetch_target_20261105/implementation/run_cuda.py) |
| 五项匹配消融源码 | [seed2027](remote_runs/ffpp_latest_dual_ablation_seed2027_20261105/) · [其余种子](remote_runs/ffpp_latest_dual_ablation_remaining_20261105/) |
| 消融的预取目标域推理 | [run_cuda.py](remote_runs/ffpp_latest_ablation_prefetch_fix1_target_20261105/implementation/run_cuda.py) |
| 三种子增强语义路径 | [semantic_half_three_seed_fix1_20261104](remote_runs/semantic_half_three_seed_fix1_20261104/) |
| CrossLayerNoImage 基座 | [iteration_02_crosslayer_fix1_20261102](remote_runs/iteration_02_crosslayer_fix1_20261102/) |
| 实现与实验协议 | [docs/protocols](docs/protocols/) |

各实验目录的 implementation 为实现，REGISTRATION.json 等文件为当时冻结的配置与注册证据。LOCAL_MANAGER.py、LOCAL_VERIFIER.py 等按原快照名称保留，属于原实验环境的调度、核验或构建归档。

## 环境与源码核验

原服务端为 Linux、Python 3.10.8、PyTorch 2.3.0+cu118；直接运行依赖见 [requirements.txt](requirements.txt)，版本记录见 [ENVIRONMENT.json](reproducibility/ENVIRONMENT.json)。该记录不等于完整环境锁。

无需 GPU 和训练数据即可核验源码：

    python tools/verify_snapshot.py

核验工具检查发布文件的 SHA256、Python 语法和 JSON 格式，不启动训练或神经网络推理。

## 复现实验

此发布为忠实源码归档，**不是已经完成路径迁移的一键训练包**。源码保留原实验的绝对路径、数据身份、注册哈希及阶段检查。

1. 按 [EXTERNAL_DEPENDENCIES.json](reproducibility/EXTERNAL_DEPENDENCIES.json) 和对应实验配置准备数据清单、对齐图像、有效掩膜、冻结语义分数与权重。
2. 在原环境恢复路径后核对文件身份；迁移机器时需同步处理路径和注册关系，单改一个路径不能保证可运行。
3. 按相应目录的配置及阶段入口执行。源码中的本地管理脚本还可能依赖原工作区构建工具，不应当作通用命令直接运行。

本次仅发布源码，不重新训练或生成新的实验结果。

## 发布范围

- 包含：完整模型、训练与推理实现、消融、语义基座、项目依赖源码、冻结配置、协议与注册记录。
- 不包含：训练/验证图像、掩膜、原始视频、模型权重、语义分数缓存，以及论文稿件和论文排版脚本。这些材料继续保存在远端备份及其原始位置。
- [SOURCE_MANIFEST.json](reproducibility/SOURCE_MANIFEST.json) 为本次代码发布清单，保留原备份中的来源和哈希；完整数据清单与 SHA256SUMS 留在远端。
- [BACKUP_VERIFIED.json](reproducibility/BACKUP_VERIFIED.json) 记录整个远端备份的验证状态，覆盖范围大于本仓库；[SNAPSHOT_README.md](reproducibility/SNAPSHOT_README.md) 为原备份说明。

原始代码包 SHA256：622d12d06a8ccfa2cb8c3e95a8f4815080549260f524afb1846d1a0bc65af907。

许可及来源见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
