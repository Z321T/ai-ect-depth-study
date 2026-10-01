# 测试侧近重复审计

2026-10-01。沿D010原算法/阈值，没有根据模型表现改变数学口径。只做数据诊断，不训练最近邻分类器或计算test分类准确率，不修改grouped_v1。

| 查询→参考 | 完整对数 | raw最小距离 | shape最小距离 | raw/shape候选对 |
|---|---:|---:|---:|---|
| test4800→train24000 | 115,200,000 | 0.0315716704 | 0.0168110199 | 0 / 0 |
| test4800→validation3200 | 15,360,000 | 0.0238729706 | 0.0168623571 | 0 / 0 |

raw是原单位联合L2差/查询联合AC范数，阈值≤0.001；shape逐通道去DC后联合unit L2差，阈值≤0.01。各4800条适用、0条N/A，无候选故不生成候选图。CSV保留最近邻身份和所有origin，类别只在无标签搜索后记录。

诊断仅打开选定两集合x/y，校验冻结summary、选定manifest/缓存SHA及标签顺序。raw原文件不在CLI重读，来源身份继承prepare元数据；完整训练读取器仍进行全量文件身份校验。

没有阈值内候选不证明采集来源独立，也未排除时间移位、裁剪、符号变化、单通道缩放。当前可报告边界是已知完整波形/名义人员组隔离，以及两类冻结距离无候选。

结果：[test→train](../results/similarity/train_test_v1/summary.json)、[test→validation](../results/similarity/validation_test_v1/summary.json)。[独立复核](../results/similarity/test_side_verification_v1.json)不导入项目算法、不用cdist或点积筛选，float64直接差值完整穷举两项的两个指标；全部4800最近索引/距离/候选数匹配，最大误差0。记录绑定summary/输入/代码SHA及验证脚本内容，原始来源未重新核验。
