# DC/AC尺度分离训练协议

本轮是原方法有限1000轮基线的探索性输入处理对照。主比较为新输入、原BN状态；固定train BN校准只作为四格消融中的另一列保留。E1校准已有负结果，不按校准后的validation重新选权重。

## 输入与训练

`src/ect/dc_ac.py`只用原单位缓存的完整250×2 clean train拟合总体DC均值、DC总体标准差和AC均方根；使用固定1e-12尺度下限。输入公式与[设计](plans/2026-10-03-dc-ac-training.md)一致：完整250点变换→float32→按原身份算法裁剪至224点。保留样本DC和相对AC幅度，不按每条信号幅度分别归一化。

ResNeXt1D-38结构、134654参数、同seed初始化、manifest标签、样本顺序、裁剪身份、Adam4e-5、batch128、每10轮固定seed10十crop概率平均验证、最高Accuracy/并列较早epoch权重选择均与原1000轮基线一致。5000/7500学习率里程碑不缩放，本轮不触发；不使用test，也不加入训练噪声。

新拟合统计另存`normalization_dc_ac.json`，记录24000样本、每通道6000000点及训练信号、train清单、cache元数据、拟合源码SHA。不覆盖原`normalization.json`或原单位缓存。每run另记录初始化参数SHA，按固定排序、名称/形状/dtype及连续CPU参数字节计算，独立核验重新生成同seed原结构初始化。

## 命令与设备切换

实现/实际验收结束后才可以登记并启动完整实验；登记拒绝仅有布尔状态的验收文件。最小学习验收不是模型成绩比较。

```bash
.venv/bin/python scripts/train_paper_dc_ac.py --seed 0 --smoke --device cpu --output results/paper_dc_ac/new_cpu_smoke
.venv/bin/python scripts/verify_paper_dc_ac.py --run results/paper_dc_ac/new_cpu_smoke --device cpu --output results/paper_dc_ac/new_cpu_proof.json
.venv/bin/python scripts/register_paper_dc_ac.py --acceptance results/paper_dc_ac/acceptance_v2/acceptance.json --output results/experiments/paper_dc_ac_v1
.venv/bin/python scripts/run_paper_dc_ac_experiment.py --registry results/experiments/paper_dc_ac_v1/registry.json --device cuda
```

训练/加载/独立核验和控制器均支持auto/cpu/cuda。CUDA需要能访问GPU的执行上下文；沙箱不可见不代表驱动不可用。CPU重载CUDA模型必须显式使用独立核验的`--allow-device-change`，分类预测仍要求逐条完全相同。

## 自动执行与产物

控制器依次运行seed0训练/核验→seed1训练/核验→seed2训练/核验，再执行三个选中权重的固定BN校准/独立核验。最后生成原输入/新输入×原BN/校准BN四格clean validation表图、全部组/类、固定首末100训练轮/首末10验证点及第1000轮。途中失败保存日志并停止后续依赖步骤，不自动改参数，也不覆盖或伪恢复没有optimizer/RNG的中断训练。

`registry.json`、三个配置和拟合统计不可变；`execution_status.json`只用于记录运行状态，单run的`progress.json`记录当前epoch。`.pt`留本地；公开预测CSV压缩为`.csv.gz`，原CSV本地保留用于核验，恢复时不得覆盖已有文件。三个seed样本SD不是置信区间，原始20个类别语义未确认，不换算深度指标。

当前实现/运行状态见[执行记录](plans/2026-10-03-dc-ac-execution.md)，最终结果须等待全部训练和独立核验后报告。
