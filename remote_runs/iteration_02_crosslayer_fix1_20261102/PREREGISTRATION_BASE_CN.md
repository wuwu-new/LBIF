# iteration_02：固定CLIP跨层信息与独立无图像残差控制

状态：独立设计稿，尚未注册、上传、预检或训练。实际冻结以新目录REGISTRATION为准。iteration_01源与七域已完整归档核验，追加完成数1/10。本轮实现/预检不计完成；只有完整三臂源和固定三权重七域核验齐备才计第二轮。

## 证据与单一假设

V22轮换相对固定macro−0.224756pp；iteration_01共同首轮后质量刷新为88.134944%，相对固定88.148579%为−0.013636pp，3/7域改善，最差CDF2−0.334277pp。不保留轮换或质量刷新，不把两项阴性干预组合。七域反复观察且用于设计，全部为自适应探索，单seed训练SD=NA。

实际full_source_v2只读取CLIP最后层patch经固定1024→192投影；V18多尺度仅在残差CNN的2/4/6出口。joint_search_v4b已有radius1局部cross-attention，不把新增注意力名称当新信息。docs60与docs56支持检验CLIP层来源，但没有证明末层压缩就是因果根因。

主假设：保持完整旧语义logit、相同图像残差与训练协议，在同容量旁路使用预指定中间层信息，能否比重复末层输入改善固定epoch10完整七域macro。层号不在目标集扫描。本轮不是Forensics Adapter/DFD-HR复现，不加入掩膜、原型、额外损失、门控、筛选或新语义训练。旧192维缓存不可还原中间层，真实远端CUDA重新编码8/16层。

资料边界：复用已归档的原始论文审读与来源指纹（literature_review_20261031/SOURCES.json、docs56/60）。本轮网页接口没有返回可引用正文，不声称新增联网全文复核或最新文献穷尽。真正的层号依据远端已安装CLIP软件源码和配置：24个block、224输入、14patch、257×1024；hidden_states[0]为编码器前输入，[k]为第k个block后。软件源码SHA ee6c3996aa6c25c9051bba1961200e58fcdc3ffda7f83120e83e63b2194ff3c1，配置SHA 8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a。静态证据在preparation_notes/iteration02_static_inventory_20261102T043550Z.json，不是模型预检。

## 固定三臂与公平归因

三臂均从seed2027新初始化独立训练10×719更新，不继承旧训练前缀；旧独立首轮失败仍为false，不重测、不改写。最后修正层零初始化，初始主logit等于旧语义。两Full臂相同全部初始参数、参数量和slot采样；不要求不同层输入训练轨迹逐位相同。

- LastLayerResidual：三路均取第24层，含原V18图像SRM残差，控制新读出参数和路径。
- CrossLayerResidual：取第8/16/24层，其他Full结构完全相同。
- CrossLayerNoImage：同8/16/24读出，独立训练，无RGB/SRM值访问。使用Linear192→32把融合patch送入同宽度、同深度的CNN块和多尺度adapter；16×16 token网格上取消两个无参数AvgPool，保留全部InvertedBlock以避免不必要的信息下采样。相同辅助出口、局部交互、修正头、损失、学习率与更新数。比Full多2272个可训练参数，不以减容量或简单把Full输入置零削弱控制；仍不声称相同函数空间/FLOPs。CUDA预检验证RGB扰动无影响、RGB梯度不存在。

各层先用同一个冻结历史semantic_query作1024→192投影并fp16保存，再按路由分别做无仿射LayerNorm192，拼接后Linear576→192无bias，权重初始化为[I,I,I]/3。读出进入原V18的s_project和局部交互，不修改旧完整语义出口。1024→192投影仍是压缩；阴性结果不能否定所有原生1024维跨层方法。

首先比较CrossLayerResidual与LastLayerResidual，判断层来源；再与独立CrossLayerNoImage比较，判断图像残差增量。若NoImage相当或更好，只能说新CLIP读出有用，不能把收益全算作双流。历史V18、OldV2和旧单语义为额外参照，不冒称与新模型严格同容量。

## 训练、精度、资源与预检

2873视频/291家族/11492原四帧；原path固定resize[.75,1]、blur[0,.4]、JPEG[65,95]两视图；方法平衡有放回采样2027+10000×epoch。batch16/workers2、AdamW主lr3e-4/滤波0.1倍、wd.01、clip1；原两视图BCE(s+delta)+0.1BCE(aux)均值+0.05delta一致性；g1/bound2/epoch10。不读源TEST、目标标签不训练，不以源结果正负筛掉七域臂。NoImage的aux来自token支路，不声称独立纯残差专家。

所有模型仅region-46:28081远端CUDA。读取hidden_states与原last_hidden_state逐位检查，完整旧语义head输出fp32；三层patch沿原精度路径先half→float，再固定投影与half保存。预检对原compact_encode末层和logit作精确比较，再对旧保存源预测/完整初始语义AUC核验，容差仍2e-7；不放宽历史审核。校验三臂实际输入敏感性、无图像控制不泄漏、梯度通路、无预检优化器更新、共享参数初始字节一致和全部冻结语义参数。

两套TRAIN RGB+三层压缩patch驻RAM，按公式约9.53GiB；每次构建隔离Python/NumPy/CPU/CUDA RNG。源VAL原RGB/第24层/logit约5.12GiB现有磁盘缓存逐SHA验证后只读复用，不复制；仅新增8/16投影约4.04GiB一次，源逐epoch按小批流式读。跨域不建全量新缓存。当前只读盘点磁盘28891107328字节空闲、cgroup25GiB/用4057100288字节、GPU空；三臂checkpoint与源/目标/归档保守另给2GiB，余八轮同规模输出估8GiB，另保留1GiB磁盘。设计估计不代替各阶段实查；运行中始终留6GiB cgroup和1GiB磁盘。若不足则留证报告，不全局清缓存、不删历史、不扩资源或排队争用他人GPU。

## 固定终点与取舍

源VAL690视频/22080帧、138真552假70家族，每epoch完整保存分数、权重/AdamW/采样SHA及梯度审计，固定epoch10主终点；best仅描述。源完成退出并全文件及33个epoch保存视频分数独立重放后，三臂epoch10均完整评估CDF1/CDF2/DFD/DFDC/DFDCP/Wild/UADFV，共6973视频223136帧。每视频32帧概率均值，旧对照逐帧SHA/数值核验后复用，新三臂全量CUDA推理。

源归档包含全部实现/配置/日志/权重/优化器/33个epoch保存视频分数；独立核验全部非缓存文件SHA和保存视频AUC/配对统计，不冒称源逐帧预测重放。新增4.04GiB VAL中间层二进制仅远端逐SHA检查并原地保留，归档保存其布局、字节数和SHA引用，避免把可重建大缓存重复传回本地；原5.12GiB VAL缓存也不复制。完整七域归档另外独立验证全部文件、帧到视频、各AUC、1000次家族配对与macro。报告每域真假及帧分母、相对单语义/OldV2/IR/V18/匹配控制/当前研究参照、改善域数、最差退化、≥85%域数、源≥95%、修复误伤及描述性区间。单seed SD=NA；不保证89.699%。CrossLayerResidual的macro不优于LastLayerResidual则不保留该层路由；若优于末层但不优于NoImage，不能宣称图像残差机制被支持。所有阴性保留，正式OldV2与历史研究参照不自动晋升。
