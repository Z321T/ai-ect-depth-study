# 本轮1000轮训练要验证什么

用户已明确先完成当前1000轮训练，并记录验证目标。三个seed继续按原执行登记各训练1000轮，再独立核验；本轮记录的是有限预算论文方法重建的学习行为和验证表现。

本分析计划记录于2026-10-02T13:38:42 UTC，训练已开始、seed0已完成90轮，部分验证成绩已见。因此它是运行中补充的探索性分析计划，不是训练前预注册，也不改变已冻结的训练或模型选择规则。机器可读问题、当时的进度/验证点快照及SHA见[analysis_plan_v1](../results/experiments/paper_baseline_v1/analysis_plan_v1/plan.json)。

## 固定实验条件

grouped_v1：24000条train、3200条validation；保留DC及既有训练统计量。ResNeXt1D-38、seed0/1/2、Adam lr4e-5、batch128，每轮随机224点裁剪。每10轮固定seed10十crop概率平均，按最高validation Accuracy及并列较早轮次选择，Macro-F1作为补充指标，不早停。原5000/7500轮衰减保持原样，本轮不会触发。

执行身份：[registry.json](../results/experiments/paper_baseline_v1/registry.json)，SHA256 `791e85504ff55f91d829ed94464e81ae53789186a0879aa27c188c7b14cc77c4`。三个run的进度保存在各自progress.json，完整训练及核验状态保存在execution_status.json。最终结果以完整report和独立核验为准。

## 五个验证问题和判读方法

| 问题 | 固定记录与比较 | 结果能支持的判断 |
|---|---|---|
| Q1：1000轮内是否持续学习，末期是否仍在变化？ | 每轮loss和在线train Accuracy；第1–100轮与901–1000轮均值/差值；第10、100、500、1000轮验证指标；首10个与末10个验证点均值/差值 | loss下降和Accuracy上升提供学习证据；末期趋势用于判断是否值得进一步检验训练预算。曲线趋平不能证明收敛，不能预测10000轮及衰减后的成绩。 |
| Q2：学会train后，是否能推广到隔离的validation？ | 选中权重在完整train/validation均使用eval、固定十crop的Accuracy/Macro-F1及两集合差值；在线train曲线另列 | 同种推理条件下的差距提供泛化诊断。差距不能单独确定过拟合、BN或分布偏移原因。 |
| Q3：重建方法比现有流程有多少变化？ | 全部三个seed的选中validation Accuracy/Macro-F1及均值/样本SD；与既有SVM、CNN、普通ResNet和增强ResNet的clean validation分别比较 | 正负差值都保留。Accuracy报告百分点差值，Macro-F1报告原尺度差值；只描述流程整体变化，不把多个组件/训练预算共同改变后的差值当单组件创新效果。 |
| Q4：变化是否稳定，是否由某个seed、组或少数类别支撑？ | 三个seed全列，样本SD(ddof=1)、min/max；validation组3/11和train各连接组指标；20类混淆矩阵、Recall/F1，包含零Recall类别 | 判断结果是否均衡或高度依赖某个seed/连接组；三个seed不能替代独立数据重复或构成置信区间，连接组不等同已确认的实际人员。 |
| Q5：结果是否可信且可重载复算？ | 每seed从保存权重独立核验train24000/validation3200预测；三seed共81600条。CSV独立直算总体/每类/组指标，沿既有1e-12容差；绑定源码/配置/数据/权重/CSV/proof SHA | 证明指标来自所登记的模型与数据。核验通过只证明执行与结果链可信，不能替代科学性能结论。失败/未核验结果如实记录。 |

既有clean validation参考：SVM Accuracy60.91%、Macro-F1约0.6086；CNN三seed平均Accuracy25.33%、普通ResNet22.84%、增强ResNet15.18%。最终差值从[既有完整训练结果](full_training_results.md)及其原始汇总读取，避免用展示值取代精确计算。旧模型训练轮数、裁剪、优化及选择规则有差异，比较中写明这些条件。SVM本身属于AI，当前对照不能回答AI相对非AI是否提升。

在线训练包含随机裁剪、train模式批统计和持续参数更新，不能直接与eval/10-crop验证Accuracy相减，称为泛化差距。Q2只使用选中权重在同种推理条件下的结果。

## 已见早期观察与待填结果

记录快照中seed0已完成90轮，最好早期validation出现在第80轮，Accuracy为34.6875%、Macro-F1为0.308614。第1轮train loss为2.9481，到第90轮为1.1824，在线train Accuracy达到58.59%；中间验证表现已有反复。以上来自已见的运行读数，不是最终选中模型或三seed汇总，也不是预先设定的性能目标。

本段保留记录计划时的状态；五个问题的完整结果已写入[1000轮结果与诊断](paper_baseline_1000_results.md)。原计划要求完整训练和独立核验后填写结论。结果表应包含每个seed的选中轮次、eval train/validation指标、Q1固定窗口趋势、Q2差距、Q3全部对照差值、Q4组/类别表现及Q5核验身份；同时导出学习曲线、连接组对比和混淆矩阵。计划记录时尚无完整结果，现已完成三seed训练与核验。

不设置“必须超过原文93.58%”的通过门槛，不由已经见过的test选方案；类别语义未确认，只报告class_index。本轮1000上限没有足够收敛的实证依据，既不证明原10000轮设置有效，也不证明它无效。训练后根据这些诊断决定下一项受控实验，不能从单个曲线形态直接宣布失败原因。
