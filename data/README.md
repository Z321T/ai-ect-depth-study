# 原始与衍生数据

`raw/MDDECT_v1_train.npy` 和 `raw/MDDECT_v1_test.npy` 是用户提供、与Kaggle官方v1 SHA256一致的原始数据。不修改，不上传公开仓库。

获取： https://www.kaggle.com/datasets/mchikyt3/mddect ，下载两个文件后放入raw/。数据许可CC BY 4.0，发布者mchikyt3；引用论文arXiv:2104.02472。

原文件的train/test名称代表发布来源，不是本项目正式划分。实际集合使用根目录 `manifests/grouped_v1/`；审计和原始来源索引位于 `results/data_audit/`。每条波形由source和五轴索引读取，原始1250×2数据不可直接覆盖为250点。

预处理衍生数据的目录在实现阶段明确，目前没有预处理缓存。
