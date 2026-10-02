# 固定权重与五次噪声重复评价

2026-10-02。沿用formal_v1冻结协议，评价入口不重训、不选epoch、不拟合统计量。新增evaluation.py/evaluation_summary.py与CLI，保持原训练/模型/噪声源码及登记不变。

每集合十个模型：九网络权重和单SVM Pipeline。Clean各一次，30/20/10dB各五个noise seed；validation10–14、test100–104，同一namespace为split名称。每个条件先由冻结AC噪声函数生成一份原单位float64信号，十模型共享；深度网络再使用固定训练标准化，SVM再提取既有25维特征和训练Scaler。报告记录共享信号SHA、实际残差SNR和不适用数量。每集合16条件，每模型全量样本。

评价前校验登记源码/模板/配置/类别映射/审计/缓存和清单SHA，九项完整训练验收记录及报告/权重/全量行身份，SVM报告/产物和全部模型类别顺序。推理调用seed_everything固定确定性、单线程、禁TF32；仅索引指定split，完整文件身份哈希/mmap按既有PreparedDataset范围允许。

每条预测保存split、run/family、training_seed、condition、noise_seed、SNR适用、row、wave_sha256、group_id和class_index。总体、每类和名义连接组指标按独立计数公式计算，非语义类别指标。汇总每训练seed内部五次扰动的均值/样本SD，再对三训练seed均值计算家族均值/样本SD，单列噪声波动；SVM训练SD为null，Clean噪声SD为null，不复制Clean或将15值作独立训练。

```bash
.venv/bin/python scripts/evaluate_registered.py --device auto --output results/evaluation/validation_v1
.venv/bin/python scripts/evaluate_registered.py --split test --device auto --acceptance results/evaluation/validation_v1/acceptance.json --output results/evaluation/test_v1
```

已有输出目录拒绝覆盖，report最后发布；异常撤销成功report，保留failure及部分产物供诊断，不视为完成。重试使用新输出目录，不改变配置。

test执行需独立验收JSON绑定当前registry、评价源码表、完整validation report及CSV/summary SHA、全量预测覆盖。代码或验证产物变化使门槛失效；test使用的模型报告/权重绑定必须与该验证完全一致。这是实现和研究流程门槛，用户已授权项目推进，条件通过后可执行，无需重复请求批准。validation分类核验及实现回归/独立审查完成之前不生成test预测；一旦执行不据test成绩调参。

评价源码SHA取实际执行模块所在目录，并要求project-root对应文件完全一致；不记录未执行副本作为代码证据。验收样本数须等于登记的validation总数，汇总须与完整160组指标重新计算一致。异常先写failure marker再尽力撤销report，即使撤销失败也明确为失败，读取端拒绝混合成功/失败状态。

增强ResNet与普通ResNet同时改变训练扰动与epoch选择目标，只比较完整流程。单试件、名义连接组和已冻结距离下的去重边界不能扩大成实际采集独立性或现场效果。全部结果以class_index报告，类别映射未确认。
