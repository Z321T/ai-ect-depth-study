# 结果与数据来源

本目录的数据审计、划分相关记录、近重复距离、SVM验证预测和统计图均由公开的[MDDECT v1](https://www.kaggle.com/datasets/mchikyt3/mddect)计算。发布者mchikyt3，许可为[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，参考论文[arXiv:2104.02472](https://arxiv.org/abs/2104.02472)。许可字段原始证据见provenance/kaggle_view.json。

用户提供的两份NPY与官方公开v1逐文件SHA256相同，比较证据见provenance/official_file_comparison.json。本项目变更包括完整重复诊断、整组去重划分、固定FIR下采样、近重复距离及训练/验证分类分析；不代表原作者训练过程或为其结果背书。

逐条输出仅包含公开数据中的零基索引、类别索引、波形SHA、名义来源轴索引和本项目计算的距离/预测，不包含真实人员姓名或用户提供的个人文档。模型二进制、原始NPY、论文/译文/旧计划及本地环境不在结果提交中。最终测试分类成绩尚未计算。
