# 训练诊断与论文组件重建实施计划

> 执行采用systematic-debugging、test-driven-development和独立审查；按任务验收，不请求已授权工作的重复确认。

**Goal:** 找到当前低分模型的具体学习行为，完成论文网络/裁剪组件的可运行重建，为新基线登记提供证据。
**Architecture:** 新诊断模块重载formal_v1，只推理train/validation；旧训练/模型/噪声源码不改。论文模型和裁剪另建模块，先用小数据检查梯度、过拟合与CPU/CUDA重载，不将小数据验收视为性能改进。
**Tech Stack:** 当前NumPy/PyTorch/scikit-learn环境；auto/cpu/cuda沿用devices.py。
**Spec:** docs/paper_baseline_alignment.md、D014。

## 全局约束

- 冻结registry/九配置/旧源码/模型/缓存/清单保持不变，保留全部seed。
- 不推理test、不改划分、不猜类别语义、不为高分挑seed。
- 新输出原子创建，已有目录拒绝；report最后发布，失败marker使混合状态无效。
- 绑定实际执行源码、登记、缓存、输入、清单和模型文件SHA，真实GPU执行用于CUDA验收。
- 本单元不声称严格复现10000epoch或达到论文成绩；新完整训练另立协议。

## Task 1：固定权重诊断

Files: 新建src/ect/learning_diagnostics.py、scripts/diagnose_learning.py、tests/test_learning_diagnostics.py。
Interfaces: diagnostic_logits(bundle, raw, batch_size=128, batch_stats=False, order_seed=None)返回按原行序logits；diagnose_learning(root, output_dir, device='auto', registry='results/experiments/formal_v1/registry.json', cache_dir='data/processed/grouped_v1')输出报告。

- [x] 测试先失败：真实BN小网络在诊断前后参数/运行统计量/训练模式相同，顺序恢复正确；非有限输出拒绝。
- [x] 实现固定eval推理和克隆模型的batch-statistics反事实；仅BN使用当前批统计，其余层eval，无梯度，无running更新。不作为可部署成绩，明确依赖同批样本。
- [x] 固定两种混合批序种子20261002/20261003、batch128；train/validation两集合、九权重各三模式；SVM只固定推理。无拟合、无重新选epoch。
- [x] 增加真实fixture完整登记集成：禁止索引test，读取模型前后文件SHA一致，输出CSV逐条还原指标，目录拒绝/失败记录。
- [x] 实际全量运行并重算指标。标准化输入报告逐通道全局统计及样本AC RMS分位数，BN运行方差摘要，学习曲线对应选中epoch/损失。
- [x] 独立审查后写docs/learning_diagnostics_results.md，假设与实测区分；涉及后续实验时另立新协议。

## Task 2：论文结构核查与组件

Files: 新建docs/paper_model_reconstruction.md、src/ect/paper_models.py、src/ect/paper_crops.py、tests/test_paper_models.py、tests/test_paper_crops.py。

- [x] 从原文图1/表I确定ResNeXt1D-38层次，标记padding/BN/shortcut/概率平均等未明确细节；核查参数量与表格一致性，不默默修正文献。
- [x] 按确认蓝图先写测试，再实现四stage各三瓶颈、分组卷积、224点输入、20logits输出；实际梯度与CPU/CUDA state_dict重载。
- [x] 身份/epoch固定随机裁剪250→224，起点0..26含端点；验证固定10随机crop，平均softmax概率并标记项目选择。跨批序/设备裁剪一致。
- [x] 小数据过拟合与真实CUDA/CPU迁移验收：固定train每类2条（40条）、抽样seed20261002，模型seed0，Adam lr4e-5/betas0.9,0.999/eps1e-8，无weight decay；随机224crop，每epoch一批40，最多2000epoch；每100epoch在同40条以固定seed10做10-crop，首次Accuracy≥95%结束，否则保留达到上限的结果。完整配置/行号/源码SHA在新结果前登记。不使用validation/test分类，验收集不是独立验证，不据记忆成绩声称性能改进。

## Task 3：整体验收与记录

- [x] 基线全项目147项真实GPU回归通过，再运行新增必要测试及独立代码审查；审查发现的问题先复现后修复。
- [x] 核对冻结源码/registry/cache/weights SHA未变、公开文件不含NPY/权重/原始资料，生成图表并目视核验。
- [ ] 更新三个根记录、D015（诊断/重建实际选择），提交/发布已授权仓库并核验远端SHA。
- [x] 根据诊断与组件验收建立下一单元有限预算论文基线设计登记config/paper_baseline_v1.json；训练器实现/验收和执行源码绑定是下一单元前置，不在这些门槛完成前启动全量新方法训练。
