# 正式实验登记：formal_v1

2026-10-01。全量深度结果产生前登记config/experiment_v1.json及九份训练配置。登记和运行完成分开记录；数据审计边界仍不等于实际采集来源独立。

## 数据与固定比较

grouped_v1共享缓存：train24000、validation3200、test4800；固定FIR和干净训练mean/std，不重拟合。test侧沿D010完成test→train、test→validation，raw相对L2≤0.001、联合unit AC形状L2≤0.01，不根据分类结果改变阈值。登记校验两审计完整身份/代码/CSV SHA、穷举规模和候选0门槛；存在候选先记录判断，不自动删样本。

| 家族 | 训练seed | epoch/模型选择 |
|---|---|---|
| CNN clean | 0/1/2 | clean验证Macro-F1→Accuracy→较早epoch |
| ResNet clean | 0/1/2 | 同上 |
| ResNet增强 | 0/1/2 | Clean/30/20/10dB验证Macro-F1等权均值→Accuracy→较早epoch |
| SVM | 已冻结完整九项搜索 | 验证Macro-F1→Accuracy→较早收敛候选 |

深度参数保持D011：既有结构、交叉熵、AdamW lr0.001/weight_decay0.0001、batch128、最多30epoch、patience8；每epoch验证noise seed10。每项完整train/validation，不使用--smoke或新增调参。增强50%干净/50%均匀10–30dB，原单位AC功率校准后标准化，保留DC。

增强/干净ResNet同时改变训练增强与epoch选择目标，解释为完整训练/选择流程比较，不作“仅训练噪声导致提升”的因果声明。CNN/ResNet规模也不同。SVM复用完整基线，登记校验其权重/预测/训练行SHA及无failure marker；确定性重复不作独立训练。

## 噪声重复和指标

权重选定后验证noise seed10–14、namespace=validation；正式test seed100–104、namespace=test。四条件Clean、30/20/10dB；Clean一次，不能复制成五个独立观察。同一样本wave_sha256/条件/seed跨模型共享扰动，零AC标记SNR不适用。

报告Accuracy、Macro-F1、每类召回/混淆矩阵与名义组表现。每训练seed先汇总噪声重复，之后汇总三训练seed的均值和样本标准差，单列噪声波动；不能把3×5合成15次独立训练。固定split种子波动不是采集总体置信区间。映射未确认，只用class_index，不计算毫米容差或具名特殊类指标。

## 命令和测试封存

```bash
.venv/bin/python scripts/diagnose_similarity.py --query-split test --reference-split train
.venv/bin/python scripts/diagnose_similarity.py --query-split test --reference-split validation
.venv/bin/python scripts/register_experiments.py
.venv/bin/python scripts/train_deep.py --config results/experiments/formal_v1/configs/cnn_clean_s0.json --output results/experiments/formal_v1/cnn_clean_s0
```

已有目录拒绝覆盖，复跑使用新目录并重新登记。registry.json记录九份配置/代码/数据/审计/基线SHA及时间；训练报告的resolved config SHA须匹配登记。真实执行状态由report/failure记录，登记不是运行完成。权重留本地；克隆须复跑数据、基线和审计后登记。

device默认auto，cpu/cuda可切换并记录实际设备。当前WSL受限沙箱CUDA不可见，允许GPU访问的上下文已真实验证RTX5070Ti；普通终端按命令执行，无需为沙箱结果改模型或驱动。

正式test分类须九项完整训练/权重/预测核验完成，固定协议与评估实现验收后一次执行。禁止依据test成绩新增调参。本轮登记不自动释放test分类，完整文件身份校验及上述test数据审计仍允许。

执行补充（2026-10-02）：上述门槛通过，validation512000预测及统计验收后记录test_release_v1，已完成test768000预测及重载核验，见docs/formal_evaluation_results.md。registry保留登记时状态，不追改历史字段；当前状态见results/evaluation/execution_status.json。
