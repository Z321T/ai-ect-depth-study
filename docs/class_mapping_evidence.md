# MDDECT v1类别映射证据核查

日期：2026-10-01。结论：尚未找到0..19对应深度/Normal/Lift-off的直接映射。config/class_mapping.json保持verified=false。核查由独立只读研究任务完成，未发送作者邮件/评论。

| 官方/作者来源 | 结果及边界 |
|---|---|
| [论文v1](https://arxiv.org/pdf/2104.02472v1)，PDF第4页§IV | 确认18个缺陷类加Lift-off/Normal，第五轴为类别；无轴内顺序声明 |
| [图5](https://arxiv.org/html/2104.02472v1/All_test_set_1_figure.png)，PDF第5页 | 图例按列排列Lift-off、2.0→0.3mm、Normal；无NPY索引 |
| [图8](https://arxiv.org/html/2104.02472v1/cm_test.png)，PDF第7页 | 混淆矩阵轴排列2.0→0.3mm、Lift-off、Normal；无NPY索引，与图5展示顺序不同 |
| [arXiv作者源包](https://arxiv.org/src/2104.02472v1) | 成员为论文/模板/文献/图件，没有数据加载或映射代码；LaTeX只引入图件 |
| [Kaggle说明API](https://www.kaggle.com/api/v1/datasets/view/mchikyt3/mddect)、[文件列表](https://www.kaggle.com/api/v1/datasets/list/mchikyt3/mddect) | 版本1，摘要式描述，文件描述空，无类别字典 |
| [发布者公开仓库](https://api.github.com/users/mchikyt3/repos?per_page=100) | 六个其他项目fork，未找到MDDECT代码；搜索无命中不证明从未公开 |
| [第一作者博士论文](https://pure.manchester.ac.uk/ws/portalfiles/portal/1574770891/FULL_TEXT.PDF)，95–96页 | 标签JSON属于另一PEC厚度数据，不能用于MDDECT；全文搜索未找到MDDECT |
| [期刊页面](https://ieeexplore.ieee.org/document/9557318/) | 本次未成功取得正文/附件，PDF请求HTTP418；不能宣称期刊补充资料已排除映射 |

原论文明确的类别集合为0.3–2.0mm、间隔0.1mm的18个深度，及两个特殊类。仅有图中排列、幅值趋势、或官方文件SHA相同不足以确定NPY索引语义。

解除语义指标限制需要发布v1的官方类别顺序声明，或可追溯的作者构造/加载/绘图代码。当前不生成“候选映射”写入配置，不计算毫米容差和特殊类别指标。索引Accuracy/Macro-F1、20类混淆矩阵及统一噪声压力测试可继续实施。
