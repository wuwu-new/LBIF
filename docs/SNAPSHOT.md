# 快照来源与发布范围

[项目首页](../README.md) · [文档索引](README.md)

代码来自2026年11月9日核验的远端“代码第一版”，保留411份源码、配置、协议及实验身份文件的原始字节。目录保持冻结版本命名，便于追溯实际训练、推理与消融实现。

| 材料 | 本仓库位置 |
| :--- | :--- |
| 代码发布逐文件来源与SHA256 | [SOURCE_MANIFEST.json](../reproducibility/SOURCE_MANIFEST.json) |
| 整体远端备份核验记录 | [BACKUP_VERIFIED.json](../reproducibility/BACKUP_VERIFIED.json) |
| 原始备份说明 | [SNAPSHOT_README.md](../reproducibility/SNAPSHOT_README.md) |
| 外部依赖身份 | [EXTERNAL_DEPENDENCIES.json](../reproducibility/EXTERNAL_DEPENDENCIES.json) |
| 本次三张插图来源与PNG校验 | [图片清单](../assets/figures/manifest.json) |
| 论文表格来源与数值 | [结构化结果](results/paper_results.json) |

远端备份还包含训练数据，其验证范围大于本仓库。图像、掩膜、原始视频、权重、语义分数缓存及完整数据清单继续留在原始备份中；仓库不分发论文全文或排版脚本。

本次首页与文档根据提供的《论文_合并稿_附录精简版》及《三张插图_v8_终版排版》整理。三张插图以288 DPI导出PNG，并保留原矢量PDF；未改绘其科学内容。

原代码包SHA256：

    622d12d06a8ccfa2cb8c3e95a8f4815080549260f524afb1846d1a0bc65af907

原项目及第三方来源分别见 [LICENSE](../LICENSE) 与 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
