# AI+自动化：涡流缺陷深度分类与附加噪声研究

本项目从 MDDECT I/Q 时序信号预测 20 个类别，研究模型比较和噪声增强。数据审计和候选独立划分已完成，下一步是预处理与分类基线；尚无正式模型性能结果。

## 文档入口

- [当前任务与阶段](task_plan.md)
- [证据和待核实问题](findings.md)
- [推进日志](progress.md)
- [研究设计](docs/research_design.md)
- [决策记录](docs/decisions.md)
- [数据阶段实施计划](docs/plans/2026-10-01-data-integrity.md)
- [数据审计报告](docs/data_audit_report.md)

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

模型阶段的依赖和命令在实现后补充，不把尚不存在的训练脚本列为可运行功能。

## 审计与划分

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/build_splits.py
```

命令从项目根目录执行，输出会覆盖同路径的审计/清单；如需比较协议，使用 `--output` 指定新目录。先冻结输入和划分，再训练。NPY中的class轴直接提供类别索引，无需额外标签文件；正式深度语义尚未核实。

截止2026-10-25；目标训练资源RTX5070Ti 16GB。训练阶段将支持auto/cpu/cuda；当前数据工具仅需CPU。

## 代码管理

当前托管工作目录的 `.git` 受环境保护，本地使用另一元数据目录：

```bash
git --git-dir=.project-git --work-tree=. status
git --git-dir=.project-git --work-tree=. log --oneline
```

从GitHub正常clone后直接使用普通git。原始数据和本地环境均在.gitignore中。
