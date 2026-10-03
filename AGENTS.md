# 项目协作约定

本项目是AI+自动化涡流检测课程研究。继续工作前读取 `task_plan.md`、`findings.md`、`progress.md`，再读取当前阶段设计。每次推进更新这些记录；研究协议改变记入 `docs/decisions.md`。

用户不要求处理人员分工。截止2026-10-25，GPU RTX5070Ti 16GB，训练支持auto/cpu/cuda。受限沙箱GPU不可见；同一.venv在允许GPU访问的执行上下文已真实CUDA运算/训练/重载验收。GPU训练与真实CUDA测试使用可访问GPU的执行方式，不为沙箱修改模型或驱动。

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

SVM/近重复依赖requirements-models.txt，协议见docs/svm_baseline.md和docs/plans/2026-10-01-svm-diagnostics.md。开发训练只计算validation指标；2026-10-02正式test已按冻结协议评价并核验，不依据测试分数调参。近重复CLI只打开所选两集合，允许validation→train、test→train、test→validation；summary写明校验范围，test数据审计不计算分类成绩。SVM按manifest标签顺序核对、训练拟合Scaler，不增加基于验证表现的新网格。model.joblib不提交Git；失败记录保留，复跑新目录。

当前托管环境保护根目录 `.git`。本地Git元数据存于 `.project-git`，本地操作使用 `git --git-dir=.project-git --work-tree=.`。从GitHub正常clone的工作目录使用普通git，无需此参数。不要提交虚拟环境、原始大数据、凭据或Git管理目录。

深度模型/噪声协议见docs/deep_models.md、D011和config/deep_v1.json；依赖requirements-deep.txt，按文档选择CPU/CUDA构建。用户要求直接推进、不再评估计算资源。训练只做train/validation，输入保留DC；附加噪声先原单位AC功率校准再标准化。每epoch验证noise seed10，训练噪声按身份/epoch固定；正式test释放及完成记录见results/evaluation/。.pt不提交Git；load_deep_model核验完整产物及源码SHA。最小验收不是正式模型比较，CUDA跳过不算真实GPU验收。

formal_v1登记、九项完整训练、noise10–14验证重复及100–104正式test均完成；正式结果见docs/formal_evaluation_results.md。registry与九份配置保持不可变，其false字段是登记时历史状态；execution_status为可更新运行状态。validation512000/test768000预测已逐条重载核验，评价源代码及统计有验收记录，147真实GPU回归通过。新的模型/参数另立探索性协议和目录，不覆盖本轮；最终test已见，不能将后续改动当未见test的确认性实验。独立推理先按seed_everything配置确定性/TF32开关，再load_deep_model。

用户后续要求先建立原方法基线再做可靠改进。固定旧权重train/validation诊断761600预测已独立核验，BN反事实只诊断、不作部署成绩。论文网络/裁剪组件见docs/paper_model_reconstruction.md，新增paper_models.py/paper_crops.py，不修改旧源码以保证重载。40条train学习验收1700epoch记忆100%，CPU/CUDA重载通过；这是组件证据，不是泛化。全项目194项回归通过。结果入口docs/learning_diagnostics_results.md、docs/paper_components_acceptance.md。

当前按docs/plans/2026-10-02-paper-baseline-training.md推进。训练器paper_training.py、执行登记paper_registration.py和独立核验verify_paper_run.py完成，终版264项真实GPU回归、CPU/CUDA同40/40两轮及CUDA权重CPU迁移共240条预测独立核验通过。验收见results/paper_training/runner_acceptance_v1/；执行registry/configs见results/experiments/paper_baseline_v1/，13份源码/数据/验收已冻结，不修改绑定源码与配置。接续三个seed全量train/validation；可变状态见execution_status.json，各run的progress.json只是进度，完整report发布及独立核验后才报告最终成绩。config/paper_baseline_v1.json为有限1000epoch设计；保留原5000/7500学习率衰减不缩放，在当前上限内不触发，不能称10000epoch严格复现。原文38层参数表与公开结构计数不一致，按公开结构134654实现，所有未明细节明确登记。不处理人员分工，不重新评估资源。

公开评价/诊断预测CSV以predictions.csv.gz保存，未压缩CSV本地存在但忽略Git；按compression.json及报告SHA核对，还原时拒绝覆盖。绘图工具scripts/plot_evaluation.py只读完成报告/汇总，不执行推理。新完整训练与改进为探索性，不使用已见test挑方案。

用户已确认先完成当前1000轮三seed，不立即启动10000。验证问题/固定分析窗口见docs/paper_baseline_1000_questions.md、D017及results/experiments/paper_baseline_v1/analysis_plan_v1/；该分析补充时seed0已90轮、早期validation已见，不称训练前预注册。五项问题待完整训练/独立核验后填写，保留全部seed与较差结果。受限上下文可能看不到实际GPU上下文PID，进程存活核对在相同授权上下文执行，不因沙箱ProcessLookupError重启训练。

2026-10-03更新：paper_baseline_v1三seed均1000epoch及CUDA独立核验完成（81600预测/全部指标误差0），analysis_results_v1/和docs/paper_baseline_1000_results.md记录五项问题、所有seed/组/class及图/时长。平均validation Accuracy72.3125%、样本SD8.7774pp，SVM差+11.40625pp；不当单组件创新或原文严格复现。选中轮800/820/770，seed1训练大组1/2明显较低而其他组较高，原因未定，下一步另立受控协议，不启动新训练。最新全项目264项真实CUDA回归通过51.253秒无跳过；原13源码/登记/config/数据/权重保持不变。

下一阶段入口docs/plans/2026-10-03-controlled-followup.md、D018：先实现E1固定选中权重、全部train-only BN运行统计重校准与配对核验，再E2保留DC的尺度分离/1000轮同预算消融，E3另立10000轮原预算及候选同预算对照，E4固定噪声/报告。新实验尚未执行，设计不等于执行登记。旧checkpoint没有末轮/optimizer/RNG，不能等价续跑9000轮；原冻结源码/配置/结果不改写，已见test不参与方案选择。人员分工/资源估算不处理。
