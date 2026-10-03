# 下一单元：有限预算论文方法基线

2026-10-02。配置为config/paper_baseline_v1.json；设计登记时训练器尚未实现/绑定，历史状态保持不变。当前已实现训练器及登记/独立核验工具，终版264项真实GPU回归、CPU/CUDA及独立核验已通过，执行registry已冻结，三个run均已完整1000轮训练和独立核验，结果见docs/paper_baseline_1000_results.md。前置为docs/paper_model_reconstruction.md及组件实际学习/重载验收。

设计身份见results/paper_components/baseline_design_v1.json，绑定配置/组件源码/数据/验收SHA并显式记录training_runner_implemented=false、execution_registry_bound=false。下一单元另创建完整执行登记，不改写这个历史设计状态。

## 固定研究条件

用户已明确先完成本次1000轮并记录验证目标，见[五项诊断问题与固定判读口径](../paper_baseline_1000_questions.md)。此分析补充于训练启动、早期验证已见后，不作训练前预注册；当前训练/选择配置不变。

使用grouped_v1全部24000train/3200validation及既有训练统计量，保持DC。ResNeXt1D-38，seed0/1/2；Adam lr4e-5、betas0.9/0.999、eps1e-8、无weight decay、batch128，最多1000epoch、不早停。每epoch身份固定随机224crop；验证固定seed10、随机10-crop概率平均，每10epoch验证一次，按Accuracy最高、并列较早epoch选择。Macro-F1及每类/组指标记录但不另改变选择目标。

原文10000epoch与5000/7500衰减不改写；本轮只执行它的初始学习率阶段，1000为提前固定的有限预算上限，不把衰减点缩到500/750，不称完整严格复现。该上限属于实验规则，不重新做计算资源估计。记录所有失败/seed，不由现有测试成绩调整预算。

## 训练器实现与验收

新增src/ect/paper_training.py和scripts/train_paper_baseline.py，不修改原deep/models/noise；支持auto/cpu/cuda及CPU保存/重载。训练接口只允许train/validation，无test模式。逐epoch保存训练损失、online训练准确率及BN摘要，固定验证interval；运行目录拒绝覆盖，失败记录保留。记录并绑定实际执行源码、配置、缓存、清单、数据与checkpoint身份，按确定性配置重载。

必要回归先写：标签/行号对应、train/validation隔离、裁剪在epoch内可复现、optimizer与非缩放schedule一致、interval/tie选epoch、checkpoint/config/source篡改拒绝、失败/成功混合状态拒绝。小夹具真实CPU训练与GPU最小路径、十crop逐条独立重载及CSV指标核验完成后，冻结执行registry，再开始三个完整训练。

完整结果只计算validation，另外对选中权重计算train作为诊断。学习行为验收可信后再固定噪声验证与改进对照；已见test不参与方案选择，后续全为探索性。不能将40条训练子集100%记忆成绩写成分类研究结果，也不能将基线重建带来的变化算作创新方法提升。

## 阶段结束标准

训练器及登记验收、三seed完整训练、逐条推理/统计核验、收敛与分组图表、更新长期记录/公开代码与派生结果。此前没有完整基线成绩，既有SVM/CNN/ResNet结果继续保留。

## 执行检查点

- [x] 新训练器必要测试先失败，再实现训练/验证、原schedule、选择及CPU产物链；十二项训练器回归在实际GPU可访问上下文全部通过。
- [x] 执行登记器及独立推理/CSV核验工具实现、必要回归和独立审查通过。
- [x] 同数据固定小规模CPU/CUDA训练、十crop独立重载及全部指标验收，形成源码绑定的runner_acceptance。
- [x] 冻结三个全量执行配置/registry，在所有结果前记录登记时间与SHA。
- [x] 按登记完成seed0/1/2全量训练，保留所有轮次与失败状态。
- [x] 独立重载全部预测/指标，生成曲线/分组图及结果说明，独立结果复核通过。
- [ ] 发布已授权仓库并核对远端。

执行冻结身份：`results/experiments/paper_baseline_v1/registry.json`，SHA256 `791e85504ff55f91d829ed94464e81ae53789186a0879aa27c188c7b14cc77c4`。验收见`results/paper_training/runner_acceptance_v1/acceptance.json`；历史design不改写。
