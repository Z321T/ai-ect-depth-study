# 论文模型BN重校准：实现与复跑协议

2026-10-03。对应D018及[受控设计E1](plans/2026-10-03-controlled-followup.md)。本文件说明实现接口；实际完成状态以新实验registry、完整report及独立核验为准，不以文档或smoke成绩作为完整性能证据。

`src/ect/paper_bn.py`和`scripts/recalibrate_paper_bn.py`是独立模块；原训练器、网络、裁剪、数据和旧登记保持不变。原40/40两轮验收权重用于新模块CPU/CUDA最小验收，不重新训练它们。完整实验复用旧1000轮各seed选中权重。

## 固定干预与评价

仅复制模型并重估BN缓冲，包含BN仿射参数在内的可学习参数逐值不变。训练数据按原行表顺序单遍遍历，全量必须为manifest全序；seed0/epoch0随机224裁剪，batch128，保留尾批。各BN重置统计、momentum=None累计批平均，之后恢复momentum=.01并用eval推理。完整校准188批，尾批64条；不是按样本数加权的精确全体方差。

校准只接受train，不使用validation/test的统计。输入归一化保持旧train拟合值。评价原选中epoch、固定seed10随机十crop的平均概率，train和validation使用同一推理模式；不重新按校准成绩选checkpoint，不搜索批序/校准种子/轮数。结果包含Accuracy、Macro-F1、混淆矩阵、每类及每组。

## 命令与执行门槛

在项目根目录，可用以下接口执行最小验收；输出路径必须不存在。新权重不提交Git。

```bash
.venv/bin/python scripts/recalibrate_paper_bn.py run \
  --origin results/paper_training/runner_acceptance_v1/cpu \
  --output results/paper_bn/acceptance_v1/cpu --smoke --device cpu
.venv/bin/python scripts/recalibrate_paper_bn.py run \
  --origin results/paper_training/runner_acceptance_v1/cuda \
  --output results/paper_bn/acceptance_v1/cuda --smoke --device cuda
.venv/bin/python scripts/verify_paper_bn.py \
  --run results/paper_bn/acceptance_v1/cpu \
  --output results/paper_bn/acceptance_v1/cpu_verification.json --device cpu
```

独立核验器手工实现裁剪/标准化及校准回放，不调用生产校准或预测函数；逐值检查仅允许BN运行缓冲改变，同设备回放必须精确，跨设备允许预先固定的浮点误差但类别必须一致。核对原权重、原报告、行表、缓存/清单/来源、源码、全部预测以及总体/每类/组指标。证明仅以新路径原子发布。

CPU/CUDA两个实际40/40校准与独立证明、CUDA权重CPU迁移、必要测试/代码审查通过后，形成绑定产物的`acceptance.json`。登记器校验实际报告、权重和证明语义，不接受只有成功布尔值的任意JSON。

```bash
.venv/bin/python scripts/recalibrate_paper_bn.py register \
  --acceptance results/paper_bn/acceptance_v1/acceptance.json \
  --output results/experiments/paper_bn_recalibration_v1
.venv/bin/python scripts/recalibrate_paper_bn.py run \
  --origin results/experiments/paper_baseline_v1/resnext_s0 \
  --output results/experiments/paper_bn_recalibration_v1/resnext_s0 \
  --registry results/experiments/paper_bn_recalibration_v1/registry.json --device cuda
.venv/bin/python scripts/verify_paper_bn.py \
  --run results/experiments/paper_bn_recalibration_v1/resnext_s0 \
  --output results/experiments/paper_bn_recalibration_v1/verification/resnext_s0_cuda.json \
  --device cuda
```

完整顺序为s0→独立核验→s1→核验→s2→核验，全部运行/失败保留。受限沙箱GPU不可见时使用已验证的GPU上下文，不改变设备/驱动或模型。CPU/GPU的选择仅改变执行设备，不改变校准协议。

## 产物与解释

每run写`model.pt`、train/validation行表和预测CSV、`report.json`。新report绑定原权重/选中epoch、旧基线指标、新校准配置、数据/源码及产物SHA。运行中/失败状态拒绝加载；同目录拒绝覆盖，原权重不变。完整registry包含三个seed、原完整核验证明、数据身份、CPU/CUDA验收和16份源码。

均值/种子SD和最差seed须与配对变化一起报告，不能只报最好seed。改善train却降低validation时，不当作可部署收益；即便改善验证，也只是固定选中权重的一次后处理效果，不能由此解释全部训练波动。类别映射未确认，指标仍为class_index分类。新实验为探索性，不使用已见test选型；DC/AC训练对照和10000轮预算另立协议。
