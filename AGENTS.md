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
```

预处理依赖requirements-data.txt；完整缓存及协议见docs/data_pipeline.md。prepare默认拒绝覆盖已有缓存；复跑用新--output目录及--report路径。ManifestDataset/PreparedDataset使用with或close释放映射，便于Windows交接。

当前托管环境保护根目录 `.git`。本地Git元数据存于 `.project-git`，本地操作使用 `git --git-dir=.project-git --work-tree=.`。从GitHub正常clone的工作目录使用普通git，无需此参数。不要提交虚拟环境、原始大数据、凭据或Git管理目录。
