# 代码导航

[项目首页](../README.md) · [文档索引](README.md)

源码目录名按冻结实验快照保留，以下入口用于阅读定位。

| 模块 | 主要源码与配置 |
| :--- | :--- |
| 完整局部模型、块级分类与图内对比 | [model_cuda.py](../remote_runs/ffpp_patch_region_full_20261105/implementation/model_cuda.py)、[patch_config.py](../remote_runs/ffpp_patch_region_full_20261105/implementation/patch_config.py) |
| 完整训练及原始推理 | [run_cuda.py](../remote_runs/ffpp_patch_region_full_20261105/implementation/run_cuda.py)、[plan.py](../remote_runs/ffpp_patch_region_full_20261105/implementation/plan.py) |
| 完整模型预取目标推理 | [run_cuda.py](../remote_runs/ffpp_patch_region_prefetch_target_20261105/implementation/run_cuda.py) |
| 五项消融，seed2027 | [implementation](../remote_runs/ffpp_latest_dual_ablation_seed2027_20261105/implementation/) |
| 五项消融，seed3407与4099 | [implementation](../remote_runs/ffpp_latest_dual_ablation_remaining_20261105/implementation/) |
| 消融预取目标推理 | [implementation](../remote_runs/ffpp_latest_ablation_prefetch_fix1_target_20261105/implementation/) |
| 三种子增强语义路径 | [implementation](../remote_runs/semantic_half_three_seed_fix1_20261104/implementation/) |
| CrossLayerNoImage基座 | [crosslayer_cuda.py](../remote_runs/iteration_02_crosslayer_fix1_20261102/implementation/crosslayer_cuda.py) |
| 其他已归档项目依赖 | [external_code](../external_code/) |

各实验目录内的REGISTRATION、EXTERNAL_LOCK和阶段核验JSON用于记录原实验身份。LOCAL_MANAGER、LOCAL_VERIFIER等归档脚本保留原名称与依赖，不是通用启动接口。

阅读顺序建议：先看完整模型与patch_config，再看run_cuda中的训练/推理阶段，随后用对应plan及注册记录核对实验配置。
