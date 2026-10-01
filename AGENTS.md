# 项目协作约定

本项目是AI+自动化涡流检测课程研究。继续工作前读取 `task_plan.md`、`findings.md`、`progress.md`，再读取当前阶段设计。每次推进更新这些记录；研究协议改变记入 `docs/decisions.md`。

用户不要求处理人员分工。截止2026-10-25，目标GPU RTX5070Ti 16GB，训练须支持auto/cpu/cuda。当前环境GPU访问受限，不把目标GPU当作已验证可用。

原始NPY位于data/raw/，PDF/译文/旧计划位于reference/，不改写也不纳入代码仓库。数据来源和SHA256见 `docs/data_audit_report.md`。原发布train/test重叠，必须使用新清单，不重新混入原划分。类别语义未确认，状态见 `config/class_mapping.json`；不能凭幅值猜测或计算未经确认的深度指标。

常用命令（项目根目录）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/build_splits.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/diagnose_data.py
.venv/bin/python scripts/diagnose_similarity.py
.venv/bin/python scripts/train_svm.py
.venv/bin/python scripts/train_deep.py --model cnn --device cpu --smoke --output results/deep/new_smoke
```

预处理依赖requirements-data.txt；完整缓存及协议见docs/data_pipeline.md。prepare默认拒绝覆盖已有缓存；复跑用新--output目录及--report路径。ManifestDataset/PreparedDataset使用with或close释放映射，便于Windows交接。

SVM/近重复依赖requirements-models.txt，协议见docs/svm_baseline.md和docs/plans/2026-10-01-svm-diagnostics.md。所有开发运行只计算validation指标，test分类保持封存。诊断CLI只打开train/validation缓存，校验范围明确写入summary。SVM按manifest标签顺序核对、训练拟合Scaler，不增加基于验证表现的新网格。model.joblib不提交Git；失败记录保留，复跑新目录。

当前托管环境保护根目录 `.git`。本地Git元数据存于 `.project-git`，本地操作使用 `git --git-dir=.project-git --work-tree=.`。从GitHub正常clone的工作目录使用普通git，无需此参数。不要提交虚拟环境、原始大数据、凭据或Git管理目录。

深度模型/噪声协议见docs/deep_models.md、D011和config/deep_v1.json；依赖requirements-deep.txt，按文档选择CPU/CUDA构建。用户要求直接推进、不再评估计算资源。训练只做train/validation，输入保留DC；附加噪声先原单位AC功率校准再标准化。每epoch验证noise seed10，训练噪声按身份/epoch固定；formal test仍封存。.pt不提交Git；load_deep_model核验完整产物及源码SHA。最小验收不是正式模型比较，CUDA跳过不算真实GPU验收。
