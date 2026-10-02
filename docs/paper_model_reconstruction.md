# ResNeXt1D公开结构重建

2026-10-02，组件协议paper_components_v1。依据原文Fig1(c)、§II与TableI最后一列；这是一套可追溯重建假设，不等于作者代码或严格数值复现。保持grouped_v1、原FIR缓存与训练统计量，开发只用train/validation。

## 固定结构

输入[B,2,224]。stem为Conv3/s1/groups1，2→6，BN/ReLU；MaxPool3/s2得到长度112。四stage宽度10/20/40/80，输出20/40/80/160，每stage三个瓶颈。瓶颈主支路Conv1→分组Conv3(groups5)→Conv1；stage1首块stride1，其余stage首块在中央Conv3用stride2，输出长度112/56/28/14；四个首块均有原始输入的Conv1投影，后两块identity。

仅全网首瓶颈省略输入BN/ReLU，其余采用BN/ReLU前激活；中间两处BN/ReLU均保留；残差相加后无ReLU。末尾BN160/ReLU/GAP/FC20，forward输出logits，softmax在评价时做。

接口PaperResNeXt1D(num_classes=20, blocks_per_stage=3)，blocks_per_stage=2仅作为26层参数核查用。层顺序/步长与shortcut来自Fig1(c)的14层单元延伸，38层新增块依照§II规则，原文没有单独画38层完整图。

## 明确登记的工程选择

- padding采用TensorFlow SAME对齐：k3/s1左右各1；偶数长度k3/s2左0右1；池化补负无穷。论文没有明确padding名称，不称作者确认。
- 全部卷积/FC bias开启；Conv/Linear Xavier uniform，bias零；BN gamma1/beta0。
- BN eps1e-3，PyTorch momentum0.01，采用PyTorch实际running variance更新语义；不能声称与未知TensorFlow版本逐位等价。
- 随机裁剪250→224，identity/epoch固定CPU RNG，起点0..26含端点。验证有放回随机取10crop，平均各crop softmax概率；原文“平均输出”没有代码级口径，概率平均是本项目选择。
- 原论文Adam lr4e-5、batch128、10000epoch及5000/7500衰减为完整训练参照。当前单元仅结构/梯度/小样本验收，有限预算完整基线另立登记。

## 参数表差异

按以上公开层蓝图，26层93754参数（与原表9.38e4舍入匹配），38层134654参数（约1.35e5）。原PDF和HTML TableI都报38层1.35e6，十倍差距不能由卷积bias或BN参数计数解释。表中数量级错误是强推断，未获作者勘误确认，也可能实际实现与表列结构不同；两种数值并列保留，不为凑参数量人为加宽。

依据：[Fig1/PDF第3页](https://arxiv.org/pdf/2104.02472v1#page=3)、[TableI/PDF第6页](https://arxiv.org/pdf/2104.02472v1#page=6)、[§V/同版本原文](https://arxiv.org/html/2104.02472v1)。独立研究员用逐层公式和PyTorch meta计数核查，组件实现后主线再核验。
