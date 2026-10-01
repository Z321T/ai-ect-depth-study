# 测试侧数据审计、CUDA验收和正式实验登记

本单元延续已验收模型/噪声协议，用户要求继续推进，并提供GPU可用证据。人员分工不纳入文档，不再评估计算资源。

1. 同一.venv内外对照CUDA访问：沙箱内不可用，沙箱外RTX5070Ti实际张量运算成功。按系统调试流程记录原因，不更改训练设备逻辑或系统驱动。
2. 并行扩展近重复诊断CLI（主线负责GPU验收与实验登记）：显式query-split=validation/test、reference-split=train/validation，只允许validation→train、test→train、test→validation；默认保持validation→train。距离与阈值、中心化、零AC、tie算法完全不变，src/ect/similarity.py不修改。仅打开选定集合，CSV保留身份和origin，summary明确数据读取范围，无分类成绩。
3. 真实CUDA执行已有模型/设备测试；运行CNN clean、ResNet clean、ResNet增强三组同样200/200两epochsmoke，独立验证GPU保存、同GPU重载和CPU迁移差异，原CPU运行不覆盖。
4. 完整test4800→train24000和test4800→validation3200数据审计；独立直接差值复核最近邻/候选计数。候选不直接删数据或调整阈值；如有候选，先记录并判断再启动正式比较。不据模型结果改变数据审计口径。
5. 观察全量深度模型结果前登记formal_v1：固定grouped_v1、现有结构/参数/选择规则、训练seed0/1/2、Clean/30/20/10dB、验证noise10–14、正式test noise100–104。SVM沿既定全量选择，不把其确定性重复种子当独立训练。登记清单、配置/模型代码SHA、audit身份、运行目录和测试封存门槛。
6. 完成审计后开始完整train/validation开发训练；首个CNN seed0完成登记链路核验后，继续其余八份同一冻结配置，不用test分类进行调参。多种子批量任务与最终测试执行状态逐项记录，不把登记当运行完成。
7. 必要回归、独立代码审查、更新根记录和CUDA状态文档；发布到已授权公开仓库，不上传原始文件/缓存/权重。

完成标准：真实GPU路径有运行证据；测试侧两项冻结算法审计完成且边界清楚；正式配置在全量深度结果前登记；九项完整运行均完成并核验保存权重/全量预测；所有状态与SHA可追溯。固定噪声重复和正式test评价属于下一工作单元，不将epoch选择seed10结果冒充重复实验。

执行结果：九项完整训练/重载与逐条件预测、CSV直接指标和组指标已核验；GPU/CPU各115200预测，跨设备分类差0，逐元素容差差异如实保留。汇总和图见docs/full_training_results.md，最终test分类仍封存。
