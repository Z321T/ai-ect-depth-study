# SVM开发验证协议

日期：2026-10-01。状态：已实现训练/选参/重载；结果属于开发验证，最终测试分类指标未计算。

## 复跑

已按README生成grouped_v1缓存后：

```bash
uv pip install --python .venv/bin/python -r requirements-models.txt
.venv/bin/python scripts/train_svm.py --output results/baselines/svm_pilot_v1
.venv/bin/python scripts/train_svm.py --full-train --output results/baselines/svm_full_v1
.venv/bin/python scripts/plot_svm.py --run results/baselines/svm_full_v1
```

已有运行目录拒绝覆盖；复跑给新的--output名称。CLI默认project-root/config/cache由脚本位置解析，可从其他工作目录运行。Windows使用.venv/Scripts/python.exe等实际解释器路径。SVM当前仅CPU，默认单线程；CNN/ResNet的auto/cpu/cuda另实现，不把目标显卡当作已验收设备。

## 特征和选择

输入为250×2原单位缓存，调用extract_features得到25维float64。每通道均值/std/峰峰值/10、50、90分位数/平均绝对AC幅值/一阶差分RMS/偏度/超额峰度；另外AC协方差、相关系数、I/Q路径长度、首末距离、闭合有向面积。保留直流信息，常量通道标准化矩/相关为0；面积是闭合时序轨迹的几何描述，不解释为校准物理量。特征定义和顺序固定在src/ect/features.py。

每个候选独立拟合StandardScaler和分类器，Scaler仅访问当前训练样本。pilot是训练每类随机100条（固定seed20261001），共2000条；full使用24000条，二者验证均为3200。train_rows.json保存原manifest行序索引，标签逐条与manifest核对。

九项候选见config/svm_v1.json。LinearSVC使用dual=False、squared_hinge和OVR；RBF SVC使用libsvm/OVO。不拟合概率，scikit-learn1.9采用支持的默认无概率校准方式。两类算法比较不能只归因于核函数。验证Macro-F1优先、Accuracy次之、并列选较早候选；收敛警告/fit_status失败的候选不选。求解器在这些配置下通常确定，变更seed不等于独立随机训练，pilot种子主要控制子样本。

实现依据：[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)、[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)、[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。

## 产物与追溯

results/baselines/<run>/含report.json、train_rows.json、validation_predictions.csv和model.joblib。模型文件在Git中忽略，普通clone后需要复跑生成；报告、索引、验证预测和图保留。report记录配置及其SHA、输入/缓存/清单/特征代码SHA、环境/实际线程、各候选计时/警告/指标、选择规则、训练Scaler、验证组结果与三份产物SHA。model_roundtrip_verified表示重载预测逐条一致，不表示跨机器浮点结果必然一致。

load_svm_model(run_dir)核验成功状态、完整产物校验表及当前特征代码，再读取本项目产生的模型。报告保留源码原始字节SHA，额外使用仅统一CRLF→LF的feature_source_sha256_lf用于兼容重载；实际代码改动仍被拒绝。.gitattributes固定仓库文本LF，避免Windows原生Git检出转换manifest/CSV字节。当前Linux环境未遇到实际换行故障；此修复针对跨平台交接，不表示Windows原生验收已完成。

调用方以同一extract_features产生输入；不直接将波形交给pipeline。模型目录中的二进制不随公开仓库发布。运行先以mkdir原子占用新目录，避免同名并发竞争；产物先写临时目录，成功report.json最后发布作为完成标志。失败运行清理成功产物后保存failure.json；模型读取拒绝成功/失败标志混合状态。

SVM产物JSON显式LF写入，保证原生Windows生成后经Git文本规范化不会改变所记录的训练索引SHA；这已通过模拟回归，但整个项目的原生Windows端到端执行仍待交接时验收。

plot_svm仅读取已保存验证结果和校验过的CSV/训练索引，不读取波形。图的provenance.json绑定运行报告、绘图脚本和图片SHA。

## 研究边界

这是固定grouped_v1的验证选参结果；pilot与full训练规模不同，不作为正式模型比较。没有测试Accuracy/Macro-F1、毫米误差、具名Normal/Lift-off指标。近重复检查只排查冻结度量，不证明人员/来源真实独立，类别语义仍待核实。后续正式测试应在CNN/ResNet、噪声条件和测试侧数据诊断协议冻结后进行。
