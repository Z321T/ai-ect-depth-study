# AI+自动化：涡流缺陷深度分类与附加噪声研究

公开仓库：[Z321T/ai-ect-depth-study](https://github.com/Z321T/ai-ect-depth-study)。

本项目从MDDECT I/Q时序信号预测20个类别，研究模型比较和附加噪声。SVM/CNN/ResNet的formal_v1完整训练与冻结test评价已完成；为建立可靠改进基线，进一步完成训练行为诊断和论文ResNeXt/裁剪组件重建及学习验收。有限预算论文方法训练器/执行登记/独立核验已通过264项真实GPU回归，三个全量配置已冻结，尚无其完整分数。类别语义仍未确认，仅报告class_index指标。

## 文档入口

- [当前任务与阶段](task_plan.md)
- [证据和待核实问题](findings.md)
- [推进日志](progress.md)
- [研究设计](docs/research_design.md)
- [决策记录](docs/decisions.md)
- [数据阶段实施计划](docs/plans/2026-10-01-data-integrity.md)
- [数据审计报告](docs/data_audit_report.md)
- [数据加载与预处理](docs/data_pipeline.md)
- [类别映射核查](docs/class_mapping_evidence.md)
- [SVM开发验证](docs/svm_baseline.md)
- [SVM实测结果与图](docs/svm_results.md)
- [CNN/ResNet、设备与噪声使用说明](docs/deep_models.md)
- [深度模型实现验收与边界](docs/deep_acceptance.md)
- [测试侧数据审计](docs/test_similarity_audit.md)
- [正式实验登记协议](docs/experiment_protocol.md)
- [九项完整训练与验证结果](docs/full_training_results.md)
- [固定噪声重复评价与测试门槛](docs/repeated_evaluation.md)
- [正式测试与噪声重复结果](docs/formal_evaluation_results.md)
- [论文方法对照与研究路线](docs/paper_baseline_alignment.md)
- [固定权重训练行为诊断](docs/learning_diagnostics_results.md)
- [论文ResNeXt结构与工程假设](docs/paper_model_reconstruction.md)
- [论文组件学习/设备验收](docs/paper_components_acceptance.md)
- [完整论文方法训练与重载](docs/paper_training.md)
- [本轮1000轮训练的五项诊断问题](docs/paper_baseline_1000_questions.md)
- [有限预算完整基线设计](docs/plans/2026-10-02-paper-baseline-training.md)
- [近重复诊断结果](results/similarity/train_validation_v1/report.md)

原始论文、译文和GPT旧计划在本地 `reference/` 保留，不上传代码仓库。数据从 [MDDECT官方发布页](https://www.kaggle.com/datasets/mchikyt3/mddect) 下载，将两个NPY放在 `data/raw/`，文件身份见审计报告。论文参考：[arXiv:2104.02472](https://arxiv.org/abs/2104.02472)。数据官方许可为CC BY 4.0，使用时引用数据发布者和论文。人员分工由用户另行处理。

## 当前数据风险

本地文件与Kaggle官方v1逐文件SHA256一致；原发布train/test含跨集合相同波形。审计输出保存在 `results/data_audit/`，新划分保存在 `manifests/grouped_v1/`（24000/3200/4800唯一波形）。新集合按人员连接分量隔离，组内去重，避免已知完整重复泄漏；近重复和真实采集来源独立性尚未确认。类别索引映射未确认，状态见 `config/class_mapping.json`。

## 环境

数据审计使用 Python 3.12 和 NumPy；依赖见 requirements-audit.txt。

推荐创建环境：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-audit.txt
```

SVM依赖见requirements-models.txt；深度模型依赖requirements-deep.txt，CPU/CUDA构建安装方式见docs/deep_models.md。

## 全量预处理

```bash
uv pip install --python .venv/bin/python -r requirements-data.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/diagnose_data.py
```

缓存位于data/processed/grouped_v1/，不上传Git；所有模型共用该缓存及训练标准化参数。默认拒绝覆盖已有缓存，复跑使用新的--output目录。训练信号和滤波图见results/preprocessing/。

## 近重复与SVM开发验证

```bash
uv pip install --python .venv/bin/python -r requirements-models.txt
.venv/bin/python scripts/diagnose_similarity.py
.venv/bin/python scripts/train_svm.py
.venv/bin/python scripts/train_svm.py --full-train --output results/baselines/svm_full_v1
.venv/bin/python scripts/plot_svm.py
```

诊断查询完整validation对train，两种冻结距离下无阈值内候选；不证明其他形式泄漏都已排除。所有模型仅用训练拟合、验证选参；输出拒绝覆盖，复跑使用新--output目录。模型joblib留在本地，公开仓库保留配置/结果/预测/图；克隆后复跑生成权重。SVM默认CPU单线程，训练读取器仍执行三集合缓存完整性检查，但不提取test特征或计算test分类指标。

## 审计与划分

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/build_splits.py
```

命令从项目根目录执行，输出会覆盖同路径的审计/清单；如需比较协议，使用 `--output` 指定新目录。先冻结输入和划分，再训练。NPY中的class轴直接提供类别索引，无需额外标签文件；正式深度语义尚未核实。

## 深度模型最小验收

```bash
.venv/bin/python scripts/train_deep.py --model cnn --device cpu --smoke --output results/deep/cnn_smoke_v1
.venv/bin/python scripts/train_deep.py --model resnet --device cpu --smoke --output results/deep/resnet_smoke_v1
.venv/bin/python scripts/train_deep.py --model resnet --augment --device cpu --smoke --output results/deep/resnet_aug_smoke_v1
```

每组只用train/validation各每类10条、两epoch，验证Clean/30/20/10dB与CPU权重重载；不同全量SVM比较性能。权重.pt本地保留，公开代码/配置/验收记录。复跑选择新output目录。

截止2026-10-25；RTX5070Ti 16GB。训练支持auto/cpu/cuda，CPU与真实CUDA训练/重载均通过验收；受限执行沙箱内CUDA不可见，普通GPU可访问上下文使用同一模型代码。正式登记命令见docs/experiment_protocol.md。2026-10-02已按冻结协议完成test及五噪声重复：SVM Clean60.23%/10dB5.43%，CNN三seed均值25.58%/23.71%；不据test成绩调参。逐条评价CSV以gzip发布，还原命令见正式结果文档，模型权重仍留本地。

## 代码管理

当前托管工作目录的 `.git` 受环境保护，本地使用另一元数据目录：

```bash
git --git-dir=.project-git --work-tree=. status
git --git-dir=.project-git --work-tree=. log --oneline
```

从GitHub正常clone后直接使用普通git。原始数据和本地环境均在.gitignore中。
