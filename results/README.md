# 结果与数据来源

2026-10-03新增experiments/paper_baseline_v1三个1000轮完整train/validation运行、独立81600预测核验、精确汇总与学习/组别/混淆矩阵PNG/SVG。仍属于公开MDDECT派生结果，不包含原始波形或权重，不计算新的test成绩。report的完整BN/轮次记录保持原样，二进制权重留本地。

2026-10-02新增diagnostics/frozen_learning_v1与paper_components/acceptance_v1，同样由公开MDDECT v1派生。前者只分析train/validation旧权重，BN反事实不是部署表现；后者40条train学习/CPU-CUDA记忆验收不是泛化指标。诊断CSV以gzip发布，compression.json证明逐字节还原身份；两类模型二进制仍不提交。

本目录的数据审计、划分相关记录、近重复距离、SVM及CNN/ResNet验证预测、噪声验收和统计图均由公开的[MDDECT v1](https://www.kaggle.com/datasets/mchikyt3/mddect)计算。发布者mchikyt3，许可为[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，参考论文[arXiv:2104.02472](https://arxiv.org/abs/2104.02472)。许可字段原始证据见provenance/kaggle_view.json。

用户提供的两份NPY与官方公开v1逐文件SHA256相同，比较证据见provenance/official_file_comparison.json。本项目变更包括完整重复诊断、整组去重划分、固定FIR下采样、近重复距离及训练/验证分类分析；不代表原作者训练过程或为其结果背书。

逐条输出仅包含公开数据中的零基索引、类别索引、波形SHA、名义来源轴索引和本项目计算的距离/预测，不包含真实人员姓名或用户提供的个人文档。模型二进制、原始NPY、论文/译文/旧计划及本地环境不在结果提交中。2026-10-02已按formal_v1冻结协议完成最终测试及五噪声重复，评价CSV以gzip发布、原始未压缩CSV留本地；compression.json记录原始/压缩SHA及逐字节还原验证，解压后即可对应评价报告的原始CSV SHA。
