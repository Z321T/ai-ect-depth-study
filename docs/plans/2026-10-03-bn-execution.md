# E1实施与执行记录

2026-10-03。用户已要求继续实验；采用[已确定的E1设计](2026-10-03-controlled-followup.md)，仅train/validation、三个原选中权重、固定一次train校准，不改变已学参数。

- [x] 读取长期记录/设计，创建feat/paper-bn-recalibration，基线264项真实CUDA回归通过55.562秒。
- [x] 核心与执行器必要测试先运行：缺少paper_bn模块，6个错误、1个CUDA跳过；跳过不作验收。
- [x] 实现独立模块paper_bn、执行CLI与严格新registry，旧13源码不变。
- [x] 独立核验器并行实现：手工裁剪/标准化/BN回放及预测/CSV计数，禁止调用生产校准/预测。
- [x] 必要测试与独立审查、CPU/CUDA实际40/40校准/核验及CUDA产物CPU迁移通过，冻结源身份。
- [ ] 登记三个完整实验，执行全部train校准、train/validation固定十crop、独立核验。
- [ ] 三seed/组结果与参数/缓冲差异汇总、图表、根记录与公开发布。

新实验结果只写新目录，失败保留。校准sample顺序使用原train_rows（全量时为manifest全序），seed0/epoch0单裁剪，batch128保留最后64条，累计批均值；固定BN校准方案不作参数搜索。仍保留原选中epoch；评价seed10十crop，平均概率。验收与登记完成前不开始全量干预。

终版验收results/paper_bn/acceptance_v1/acceptance.json，SHA256 9bb14f1546436c1fbfc865426ba751009152cc638e3af298803e273befa2aae2。16份源码绑定；309项真实CUDA回归65.438秒全部通过、无跳过；CPU/CUDA40/40同设备校准回放/240条三次预测核验指标误差0，同设备BN缓冲误差0。CUDA产物CPU回放缓冲最大绝对差4.11272e-5，在预设rtol2e-4/atol2e-5联合容差内，分类差0。不能将小验收成绩作为泛化证据。

审查三项P2：精简伪证明可通过、失败/运行证据未拒绝、完整输入链可被删减。均先有真实失败回归再修正，终版只读复核无剩余P1/P2。源变化前的实际小验收保留在data/processed/paper_bn_acceptance_before_chain_fix，不作终版证据。
