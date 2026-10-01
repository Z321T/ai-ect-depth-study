# 数据加载与预处理实施计划

目标：提供SVM和时序网络共用的可信数据接口，将原始1250×2缓存为250×2；标准化参数仅由干净训练集计算。
设计依据：[研究设计](../research_design.md)，用户已授权继续推进。使用Python/NumPy/SciPy，CPU执行。

## 固定协议

- 输入：grouped_v1三个清单；类别只使用索引，不推断语义。
- FIR：101个系数，Kaiser窗口beta=5，cutoff=250 Hz（firwin半幅点），原采样2500 Hz；resample_poly(up=1,down=5,axis=-2,padtype=line)。保存全部系数和版本。
- line边界延拓避免约1单位直流信号在零填充边缘产生大幅假波形。该选择为本项目协议，不宣称与原论文相同。
- 缓存float32未标准化波形及int64标签；噪声在原单位缓存上加，再应用训练统计量。数据缓存不提交Git。
- 逐通道总体标准差(ddof=0)；批次合并使用稳定均值/中心二阶矩；禁止零标准差静默除零。
- 原数据和清单启动时校验SHA256；加载代表索引时复验波形哈希；模型不重新划分。
- 输出绑定输入/清单/处理代码/依赖版本、缓存SHA256与统计量。失败不留下看似完整的缓存，不覆盖已有输出。

## T1 清单加载

文件：src/ect/dataset.py、tests/test_dataset.py。
- [x] 构造实际七维NPY和三份清单，测试指定来源/索引读取、篡改原文件/清单被拒绝、代表标签不一致/实际哈希不一致被拒绝。
- [x] 观察测试因接口缺失而失败。
- [x] 实现ManifestDataset(project_root, manifest_dir)，records[split]和iter_batches(split,batch_size)→原单位波形[B,1250,2]、标签[B]。
- [x] 验证所有清单间完整波形、人员组隔离；不存在路径依赖当前工作目录的隐含假设。

## T2 固定预处理

文件：src/ect/preprocessing.py、tests/test_preprocessing.py。
- [x] 测试恒定直流、通道保留、50Hz保留、600Hz抑制、错误shape和非有限拒绝；测试高直流小变化统计稳定、不同batch分组一致、零方差拒绝。
- [x] 实现downsample_iq、fit_channel_stats、standardize，明确形状、dtype、ddof。
- [x] 运行测试，确保真实波形输入不变、训练标准化零均值/单位方差。

## T3 全量缓存与证据

文件：src/ect/prepare.py、scripts/prepare_data.py、tests/test_prepare.py、requirements-data.txt。
- [x] 小型集成测试：仅训练拟合统计；验证/测试偏移不影响；manifest顺序与缓存标签一致；重复输出拒绝。
- [x] 实现prepare_dataset，先写临时目录，成功后发布缓存。JSON元数据与normalization保存在缓存，实际全量运行记录保存results/preprocessing/。
- [x] 全量运行24000/3200/4800，校验shape、finite、类别计数、训练标准化；记录耗时/大小/过滤频响。
- [x] 信号可视化仅使用训练数据；模型阶段另做独立近重复诊断，不把图片当作类别语义证据。

## T4 审查和交付

- [x] 请求独立代码审查，修复实质问题，记录tests和全量检查。
- [x] 更新设计/README/根目录三个记录；提交、合入main并推送已授权公开仓库。
- [ ] 下一阶段：统计/几何特征SVM基线和CPU最小深度训练，不因未知类别语义停止索引分类工作。

验收：32项测试通过；独立两轮审查与三项修复见docs/code_review.md；修复后全量检查成功，CPU1.63秒。提交和远端状态以progress.md最后的发布检查点为准。
