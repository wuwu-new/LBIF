# 复现说明

[项目首页](../README.md) · [代码导航](CODE_MAP.md) · [实验协议](EXPERIMENTS.md)

## 核验代码

    git clone https://github.com/wuwu-new/LBIF.git
    cd LBIF
    python tools/verify_snapshot.py

核验工具只使用Python标准库，检查411份原快照文件的SHA256以及仓库中Python、JSON格式，不导入模型或读取数据集。

## 原运行环境

Linux、Python 3.10.8、CUDA 11.8、PyTorch 2.3.0+cu118。直接依赖版本见 [requirements.txt](../requirements.txt) 与 [ENVIRONMENT.json](../reproducibility/ENVIRONMENT.json)；后者不是完整环境锁。

在匹配的Linux/Python/CUDA环境中安装已记录直接依赖：

    python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

## 数据、缓存与权重

训练/验证图像、有效掩膜、原始划分清单、冻结语义分数、CLIP与增强语义权重、局部模型权重不随本仓库分发。所需原文件及哈希见 [EXTERNAL_DEPENDENCIES.json](../reproducibility/EXTERNAL_DEPENDENCIES.json) 和具体实验目录的外部依赖注册记录。

缺失伪造掩膜必须保留无效标记，不替换为全零监督。语义路径与局部输入应按帧身份配对，局部图像与掩膜保持几何一致。

## 恢复与运行

此版本保留原实验的绝对路径、注册哈希、选模证据和阶段完成状态，属于冻结源码归档。克隆仓库后不能直接把只读快照作为新实验输出目录。

1. 根据所选实验的plan与注册清单恢复原路径和外部文件。
2. 迁移到新环境时同步处理路径、数据身份、模块导入及注册关系，不跳过完整性检查。
3. 使用各目录run_cuda的真实阶段入口，并核对GPU资源和已有输出。LOCAL_MANAGER等脚本还可能依赖原工作区构建工具。
4. 保持相同源验证选模与固定32帧视频端点，再比较结果。

本次仓库更新未重训模型、未重新选择权重，也未生成新的目标域分数。
