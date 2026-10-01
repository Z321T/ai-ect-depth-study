# 研究发现与证据

更新日期：2026-10-01。

## 已读材料

- reference/archive.pdf：8 页，arXiv:2104.02472v1，2021-03-08。
- reference/Depth_Evaluation_ECT_Chinese_Translation.md：主要方法与表 II 数字已与原文对照。
- reference/AI_ECT_Course_Project_Plan.md：作为初稿参考；人员分工不进入后续设计。
- 原始两份NPY现位于data/raw/；路径移动，内容未改写。

## 论文事实

- 第 IV 节：同一块不锈钢试件、固定槽长 10 mm、宽 0.2 mm，18 个深度 0.3–2.0 mm，加 Lift-off、Normal。
- 30 人 × 8 角度 × 2 方向 × 5 重复 × 20 类 = 48,000 条扫描；每条 0.5 秒，1250×2。
- 第 V 节：按人员划分；1250→250；训练集统计量逐通道标准化；训练随机裁剪 224；验证/测试 10-crop。
- 表 II：ResNeXt1D-38 Top-1 93.58%，±0.1 mm 97.20%。本项目简化网络与处理不构成严格复现。
- 限制：跨人员不等于跨试件或跨仪器；分类 Lift-off 不证明任意提离变化下深度估计稳定。

## 上次会话本地实测（本次将持久化重跑）

| 项目 | train | test |
|---|---:|---:|
| shape | (27,8,2,5,20,1250,2) | (3,8,2,5,20,1250,2) |
| dtype | float32 | float32 |
| 样本数 | 43200 | 4800 |
| 唯一完整波形 | 30264 | 4664 |
| 多余重复条目 | 12936 | 136 |

- 无 NaN、Inf、全零或两通道均恒定的样本。
- 测试集 2928 条完整波形与训练文件重复，占 61%；直接字节比较已确认一例。
- 跨类别重复组为 0；训练文件跨人员重复组为 11465。
- 测试人员索引 1、2 与训练文件分别重叠 1474、1454 条。
- train 前 9 个人员索引的波形与剩余训练数据/测试数据无完整重复。
- 以上仅针对本地文件，不归因到论文或作者；尚未比较官方文件校验和。

## 未知事项

- 20 个类别索引的正式名称与深度顺序。
- 重复的来源以及实际独立采集单元是否与名义人员轴一致。
- 近重复和共享原始扫描片段情况。
- 当前提供文件与官方v1完全一致：已解决，证据见后文SHA256比较。

## P2预处理资料（2026-10-01）

- SciPy官方resample_poly文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html ，支持显式FIR系数和line边界延拓；默认零填充不适合本数据大直流偏置。
- firwin文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html ，cutoff是半幅点(-6dB)，不是严格截止带边界。
- 预处理固定101系数、Kaiser beta=5、250Hz半幅点，降采样后500Hz；边界line为项目选择，与原论文细节不等同。

## 来源访问记录

- https://arxiv.org/abs/2104.02472：上次会话核对论文身份。
- https://www.kaggle.com/datasets/mchikyt3/mddect：浏览器未取得详细说明；不可据此宣称官方元数据已确认。

### 2026-10-01公开API补充

- 通过Kaggle公开API成功取得view/list JSON，存入 results/provenance/kaggle_view.json 和 kaggle_files.json。
- 官方版本为1，2021-02-04发布，授权CC BY 4.0。文件名和大小与本地一致，但这尚不是完整内容身份确认。
- view的description只有论文摘要；list无类别说明；两者均未给出类别索引映射。
- GitHub公开仓库名称搜索MDDECT结果为空（results/provenance/github_repos.json）；不等于作者未公开代码。
- 官方v1归档已下载至/tmp（压缩79457869字节），逐个NPY解压流校验：两个本地文件SHA256均与官方相同。结果在 results/provenance/official_file_comparison.json。因此重复确实存在于当前公开v1文件，不能将这一结论扩展为作者训练过程已被核实。

## 本次可复跑审计结果

- results/data_audit/audit.json、samples.csv 全量覆盖48000条；与上次统计一致。
- 合并两个文件后32000条唯一完整波形。
- 人员连接共有12个分量：`test:1 + train:9–17`、`test:2 + train:18–26`各含10个名义人员、16000条原始记录、8000条唯一波形；`test:0`与`train:0–8`各为单人分量、1600条。
- 每个大分量去重后每类400条，每个单人分量每类80条，类别覆盖完整。
- 完整波形隔离可通过整分量划分实现；采集人员身份真实独立性仍不能由哈希确认。
- 官方发布train SHA256：f50bc6cb3d08f9972679f3876a41025aec2a4e2672147e24a6b58db007c71ac7。
- 官方发布test SHA256：865f972445ce0b9ad562336e7b6258b48d6b1d70d8d7254ff4fca77fdb951b58。

## 类别语义追加核查

- 独立核查论文原图5/8、arXiv作者源包、Kaggle公开描述/文件列表、作者公开仓库；均未找到NPY索引直接映射。
- 图5和图8类名排列不同，不能由展示排序推断NPY轴顺序。源包未提供绘图生成代码。
- 期刊正文/附件本次未取得，不能说补充资料一定没有映射。
- 详细出处与访问边界见docs/class_mapping_evidence.md；mapping保持unverified。索引分类链路继续，语义指标暂不开放。
## P2验收补充（2026-10-01）

加载/预处理/缓存实现后32项测试通过；独立审查发现并复查三项问题已修复。修复后全量CPU生成1.63秒、约62MiB；处理后32000波形唯一，三集合完整波形交集0。训练每通道600万点的固定标准化使均值约0/标准差约1，标签逐条与冻结清单一致。

训练缓存总功率/AC功率中位比81.12dB，说明直流主导，支持研究设计使用AC参考功率做附加噪声实验。AC展示只为看清微小波形，不改变训练缓存。图形不作为索引语义或来源独立性的证据。

## P3资料核验（2026-10-01）

scikit-learn官方文档核对：LinearSVC在样本多于特征时可用dual=False；此求解方式random_state不影响结果，不能把重复种子当作独立随机训练。StandardScaler只由训练样本计算均值/方差。距离平方的范数/点积公式可能数值消减，近重复候选需直接差值复算。

资料：[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)、[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)、[euclidean_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)、[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)。本轮诊断距离和阈值为项目选择，详见新实施计划，不能称为通用泄漏判定标准。

首次运行发现scikit-learn1.9.1的SVC显式probability=False也产生弃用警告；官方文档标记1.9弃用，安装签名默认为'deprecated'。改为省略该参数，保持默认不校准概率；夹具验证无训练警告且predict_proba不可用，未引入概率训练。此修复不改变网格或选择规则，产物将按新代码重跑。

## Train/validation近重复实测

全部3200条validation查询24000条train，共76,800,000对；CPU单线程搜索10.995秒/总11.213秒。原单位相对L2阈值0.001及形状unit L2阈值0.01的候选查询/候选对均为0，最小距离0.023379692/0.015377562，中位0.57725815/0.33373137。独立逐差值穷举验证全部最近索引/计数一致；阈值在观察之前固定，不因结果扩大。

没有候选不证明来源独立，时间移位/裁剪/符号/单通道缩放未排查。测试信号未读取，测试侧固定算法数据审计仍待完成。图形只在有候选时生成，本次无需候选图。完整来源/距离在results/similarity/train_validation_v1/。

## SVM开发验证实测

最终pilot2000/3200耗时3.83秒；full24000/3200耗时73.00秒（当前WSL2单线程CPU）。九项均收敛且无警告，二者选中RBF C10 gamma0.1，验证Accuracy/Macro-F1分别0.375625/0.375229及0.609063/0.608643；未计算test分类指标。全量两个验证连接组Macro-F1为0.617135/0.600031。不同训练规模只用于开发预算与链路，不能作为同条件正式比较。

已验收59项测试、独立两部分审查、当前代码/缓存/图SHA和从/tmp重载预测逐条一致。完整表/边界见docs/svm_results.md，模型二进制留本地。没有为得分增加网格或特征，不从邻近索引混淆推断未经确认的深度映射。

## 当前计算资源核验

以下为先前回答资源追问的历史记录；后续用户已要求直接推进、不再评估，执行以文末P3记录和task_plan为准。

用户询问CPU是否足够：当前WSL2可见Intel Core Ultra 7 265K，20个逻辑CPU且affinity包含20个；SVM全量九候选默认单线程共73.00秒，数据处理/诊断也已由CPU完成。因此当前任务无需GPU。CNN/ResNet尚未实现计时，不能从数据量或SVM耗时推断其正式多种子/增强实验耗时。保留标准auto/cpu/cuda设备切换，先以CPU实测每epoch成本，再根据总运行预算选择设备；GPU是可选加速资源，不列为研究成立的前提。

## P3深度模型与噪声验收

用户随后要求直接推进实现、不再计算资源。已实现CNN(54,548参数)/ResNet(242,836参数)、设备auto/cpu/cuda与原单位联合AC噪声；PyTorch2.9.1+cu128实际CUDA不可用，CPU链路已验收，不把构建版本或mock分支当GPU实测。

三组相同200train/200validation、两epoch完成真实梯度和保存重载；共2400条四条件验证预测逐条复核，直接CSV混淆矩阵/F1/准确率匹配。噪声最大SNR误差1.67e-11dB，直流保留、批序不改变扰动；零AC回归明确N/A。增强两轮114/96条符合伯努利抽样，非要求每批精确一半。105项测试通过、2真实CUDA跳过，两轮审查完成。

跨设备近边界argmax可因极小数值差改变，不能当权重损坏；已改同设备严格重载、CPU迁移差异单列。报告selected_epoch/配置摘要/结构与类别数须和checkpoint及冻结选择记录一致。读取器仍哈希/mmap全缓存作身份校验，不计算test分类/统计评价/选参。

本轮每epoch只有两个更新步骤，验证分数保留作链路证据而非模型结论；没有据此修改网络/超参数/选择规则。正式全量多种子/噪声重复、测试侧近重复审计及类别语义仍待后续。细节见docs/deep_acceptance.md、results/deep/acceptance_v1.json。

## GPU状态更正及正式实验登记

用户随后提供nvidia-smi证据。同一项目环境在受限沙箱is_available=false，在允许GPU访问上下文is_available=true并真实CUDA运算成功；之前不可用结论是上下文限制。22设备/模型检查均真实执行通过，三组GPU两epoch训练与2400验证预测复核成功，GPU→CPU最大logits误差8.94e-8/类别差0，结果不推断其他权重逐位一致。

测试侧两审计130,560,000对，每种raw/shape均0候选，最小raw0.03157167/0.02387297、shape0.01681102/0.01686236；独立直接差值全穷举最近索引/距离/候选匹配、最大误差0。只排查已冻结的距离变换，不证明采集来源独立，未计算test分类。

formal_v1在全量深度结果前冻结九份完整train/validation配置、noise重复和选择规则，登记不是训练完成。登记校验SVM必需产物完整与失败标记，遇发布后清理失败撤销成功marker；两项审查反例已回归修复。类别映射仍未确认，不增加语义指标。
