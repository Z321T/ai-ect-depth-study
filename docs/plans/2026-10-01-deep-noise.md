# CNN/ResNet与噪声实现单元

用户已授权直接推进，不再评估计算资源。此次交付可运行实现和最小验收，不启动正式测试分类。

1. 固定协议并实现噪声（NumPy）和网络/设备（PyTorch），两个模块并行，主线实现训练与结果追溯。
2. 针对科学口径先测试：交流功率、能量校准、直流保持、零功率标记、样本身份决定随机数；网络真实梯度与CPU权重重载；训练只读train/validation、选择规则与失败状态。
3. CNN：32/64/128通道，kernel=5，三个Conv-BN-ReLU-MaxPool块，GAP和线性分类器。ResNet：32通道stem(kernel=7)，三阶段32/64/128各两个basic block(kernel=3)，阶段首块stride=2（第一阶段1），GAP和线性分类器。两者输入[B,2,250]，不裁剪。
4. 噪声接口add_ac_noise(signals, sample_ids, snr_db, seed, namespace)返回(noisy_float64, info)；info包含每样本AC功率、实际噪声功率、适用标记。逐通道零均值后联合能量校准，保留输入DC。零AC不加噪，SNR不适用。身份随机种子为SHA256，禁止Python hash。augment_training(..., seed, epoch)返回同接口，50%干净，另50%均匀10–30dB，按身份/epoch独立生成。
5. 两模型默认AdamW lr=0.001，weight_decay=0.0001，batch=128，最多30epoch，patience=8；训练种子0/1/2。干净训练按clean validation Macro-F1选epoch，增强按clean/30/20/10dB四条件Macro-F1等权平均，tie依次Accuracy/较早epoch。每epoch固定验证噪声seed=10、namespace=validation，不据本轮得分调整超参数。最终多噪声重复10–14与测试100–104留在正式阶段。
6. 设备auto/cpu/cuda；auto仅依CUDA可用性选择，显式不可用cuda报错。噪声在CPU生成，之后固定训练标准化并转设备；禁用TF32/cuDNN benchmark，启用确定性模式，记录软件/设备。可复现不意味着CPU/GPU逐位一致。
7. 保存CPU state_dict和结构/标准化元数据；weights_only=True加载，map_location显式指定。目录原子占用，报告最后发布，SHA绑定数据、代码、配置、权重、预测；失败保留failure.json且拒绝重载。权重不入Git。
8. 实际运行CNN clean、ResNet clean、ResNet增强各两epoch最小验收，train/validation各每类10条，固定抽样种子20261001；与全量SVM分数不可比较。验证clean及30/20/10dB，同设备重载预测逐条一致，CPU迁移数值/类别差异单独记录。GPU可用则做真实前向/反向和跨设备重载检查；环境不可用明确记录，继续CPU验收。
9. 独立审查与回归；更新研究设计、决策、三个根记录、命令文档。正式实验仍需测试侧同阈值相似度审计及正式协议冻结；类别未确认只报告class_index。

完成标准：CPU链路与噪声科学不变量通过，3个实际运行可重载，设备分支行为验证，数据与代码身份可追溯；不把最小验收当作研究结论。
