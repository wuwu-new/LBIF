# SemanticHalf三种子完整执行

docs107规则下，四列完整七域及独立核验后选出SemanticHalf：macro90.43106494%，其余NoAux90.16663142%、UniformAux90.11537785%、BoundaryAux90.11607100%。这不是残差成功；三臂原始残差macro仅约57%，不据此重搜融合系数或训练落选方案。新目录semantic_half_three_seed_20261104，旧任务/automation暂停，ROUNDS3不变。

## 唯一模型、预算与不变量

完全复用iteration02修复版CrossLayerNoImage的结构、冻结CLIP/历史语义head、三层8/16/24投影、IR_LayerScale+ShallowMultiScale、零修正头初始化、固定token_input初始化seed2202027和训练目标。各seed变化的是可训练token修正/读出初始化和采样，不是重新训练冻结的历史语义anchor。不得将V4b的3407/4099或相同权重换推理seed冒充本方法新seed。

完整原TRAIN11492帧/2873视频/291家族，原4帧/视频；完整VAL22080帧/690视频/70家族。seed2027严格核验旧epoch10注册/文件SHA/model tensor SHA及64个原batch16分数后只读复用；新seed3407/4099各10epoch×719更新=7190，共14380保留更新。每个源预检两新seed各2个一次性更新，独立和runtime共8，不能混入保留轨迹。0…10各epoch完整源VAL，目标只epoch10，不取源best，不早停。

训练保持原full logit=s+delta，而非在loss前减半：原两视图平均[BCE(s+delta)+0.1BCE(token auxiliary)]+0.05 delta MSE，AdamW3e-4、wd0.01、clip1，batch16、workers2、原method-balanced WeightedRandomSampler有放回seed+10000epoch。像素增强严格原nuisance-v1:path固定resize/blur/JPEG，两个TRAIN视图只编码一次跨seed共享只读，编码与训练RNG隔离。CLIP、历史语义head及原投影不改变，所有模型权重加载和前向/训练仅远端CUDA。

## 缓存、预检与目标

注册前完整旧2027重放测试发现：历史残差实验SEMANTIC_VAL.csv按native配对VAL排序，与原语义VAL的排序不同；直接按行号接入会产生错误的0.4848508717 AUC。此时尚未注册/上传/CUDA预检/训练。现以prior_aligned_image_path、SHA、video/family/label/frame/source_frame全身份一一映射回原清单；原CSV、完整双向索引映射和重排后CSV都保留并注册，独立核验逐行重排关系，不改变任何分数或放宽误差阈值。只读复用本身不新增2027训练或全量神经推理。

旧VAL_DISK和iteration02 VAL_EARLY均SHA核验、只读；原清单图像SHA逐文件检查，32个分散源VAL样本新编码与旧三层cache比较，不能伪造或偷用不同语义特征。新TRAIN两RAM视图约9.54GiB，不复制三份、不保存新的大特征盘缓存；target新CLIP流式一次供两新seed共享，不能复用2027的修正delta作为新seed。2027全源/七域保存分数只读复用。完整目标6973视频223136帧/seed，新两模型446272残差为零的token模型帧前向；目标掩膜不读取，未知原始帧号不伪造。

源预检固定64原VAL样本（batch16起点0/5504/11040/16560），旧2027 logit误差≤2e-7；两新seed零头、RGB独立性、冻结anchor、梯度、两步AdamW、活跃state与step检查。每次源预检192个模型帧前向（旧2027×64加新seed各4×16：原输入、翻转占位输入、两步更新），另32源帧仅编码/cache核对；独立及runtime源预检384帧。目标预检每域16帧、三个seed（2027仅探针）共336帧/次，旧2027对历史概率误差≤2e-7，两新权重身份/有限性/RGB独立性；独立与runtime共672。所有隐含额外前向都计成本，不能修改冻结预算掩盖失败。

独立GPU非阻塞锁和OwnedCudaGuard空GPU绑定真实NVML身份；6GiB cgroup及1GiB盘余量，新输出预留2GiB。无外部GPU排队/抢占/终止，不删旧数据/缓存/权重/日志。CUDA预检须通过退出才训练或完整目标；源完成退出并独立归档核验后才目标。错误先留证，新目录独立修复，不修改运行中冻结文件或放宽2e-7。不下载模型/全套图像，本地只标准库静态/标量统计。

## 报告

目标每帧f32(logit)、f32(logit/2)，每视频32帧double sigmoid概率均值；原尺度语义是每seed必要对照，阈值0.5不调。逐seed源/七域分母、AUC/FP/FN、相对原尺度/旧Semantic/V18/V22/OrderedCLIP的差、改善域数/最差退化、源95%/目标85%/macro89.699%、家族bootstrap1000描述区间及排序修复/误伤、成本完整保存。三seed逐域与macro均值和样本SD，以及新增两seed的均值/样本SD单独报告；不选择最好三seed或跨seed更换模型。2027参与选型，当前七域反复观察；不称三个新独立确认/盲测，家族区间不代替种子SD或消除选择偏差。不自动部署、恢复旧轮或追加种子。

本文件须随准确实现/依赖/输入/预算SHA注册。尚未注册或CUDA通过的准备不算执行完成。
