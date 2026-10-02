# 下一单元：有限预算论文方法基线

2026-10-02。配置为config/paper_baseline_v1.json，当前只是训练前设计登记；训练器尚未实现/绑定，不能称已经开始或完成完整基线。前置为docs/paper_model_reconstruction.md及组件实际学习/重载验收。

设计身份见results/paper_components/baseline_design_v1.json，绑定配置/组件源码/数据/验收SHA并显式记录training_runner_implemented=false、execution_registry_bound=false。下一单元另创建完整执行登记，不改写这个历史设计状态。

## 固定研究条件

使用grouped_v1全部24000train/3200validation及既有训练统计量，保持DC。ResNeXt1D-38，seed0/1/2；Adam lr4e-5、betas0.9/0.999、eps1e-8、无weight decay、batch128，最多1000epoch、不早停。每epoch身份固定随机224crop；验证固定seed10、随机10-crop概率平均，每10epoch验证一次，按Accuracy最高、并列较早epoch选择。Macro-F1及每类/组指标记录但不另改变选择目标。

原文10000epoch与5000/7500衰减不改写；本轮只执行它的初始学习率阶段，1000为提前固定的有限预算上限，不把衰减点缩到500/750，不称完整严格复现。该上限属于实验规则，不重新做计算资源估计。记录所有失败/seed，不由现有测试成绩调整预算。

## 训练器实现与验收

新增src/ect/paper_training.py和scripts/train_paper_baseline.py，不修改原deep/models/noise；支持auto/cpu/cuda及CPU保存/重载。训练接口只允许train/validation，无test模式。逐epoch保存训练损失、online训练准确率及BN摘要，固定验证interval；运行目录拒绝覆盖，失败记录保留。记录并绑定实际执行源码、配置、缓存、清单、数据与checkpoint身份，按确定性配置重载。

必要回归先写：标签/行号对应、train/validation隔离、裁剪在epoch内可复现、optimizer与非缩放schedule一致、interval/tie选epoch、checkpoint/config/source篡改拒绝、失败/成功混合状态拒绝。小夹具真实CPU训练与GPU最小路径、十crop逐条独立重载及CSV指标核验完成后，冻结执行registry，再开始三个完整训练。

完整结果只计算validation，另外对选中权重计算train作为诊断。学习行为验收可信后再固定噪声验证与改进对照；已见test不参与方案选择，后续全为探索性。不能将40条训练子集100%记忆成绩写成分类研究结果，也不能将基线重建带来的变化算作创新方法提升。

## 阶段结束标准

训练器及登记验收、三seed完整训练、逐条推理/统计核验、收敛与分组图表、更新长期记录/公开代码与派生结果。此前没有完整基线成绩，既有SVM/CNN/ResNet结果继续保留。
