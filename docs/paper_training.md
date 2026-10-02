# 有限预算论文方法训练与重载

本单元执行docs/plans/2026-10-02-paper-baseline-training.md。配置与历史设计登记保持原样；新增训练器的验收和执行登记是完整训练的前置，不能将仅有设计登记的状态写成已训练。

## 实现口径

`src/ect/paper_training.py`仅访问train/validation分类数据；既有PreparedDataset仍校验三集合缓存/清单/原始文件身份，不计算test成绩。全量运行的每个resolved config是原config/paper_baseline_v1.json加seed0/1/2，实际设备以独立override选择，保留登记中的device=auto。完整目的必须通过执行registry校验才可占用运行目录；小数据验收目的不替代正式执行登记。

每轮模型train，身份/seed/epoch固定250→224随机裁剪，保留DC并使用既有全量训练逐通道统计量。记录epoch从1开始，裁剪RNG使用epoch-1；批序由seed、epoch-1和固定2026确定。Adam使用4e-5、betas0.9/0.999、eps1e-8、weight_decay0。原文5000/7500轮衰减点保留；本轮1000轮上限不触发它们，不早停。

每10轮在validation执行固定seed10十次随机裁剪、softmax概率平均；按最高Accuracy、并列较早epoch选择。Macro-F1只记录，不用于改变选择。验证在独立模型副本上eval、不更新原BN。每轮保存训练交叉熵、在线准确率、各BN统计及实际时长；仅最终选中模型计算丰富的每类/组指标。

选中模型同时对train进行固定十crop诊断，此时使用与validation相同的validation命名空间/seed10，不再优化。该结果来自参与优化的样本，只能用于学习诊断；在线训练准确率还受到随机裁剪、批统计和持续更新影响，二者不能混称。group是连接组，实际采集人员身份未确认；类别仅为class_index。

## 产物与失败处理

运行目录原子占用，禁止覆盖。`progress.json`是可更新进度，不是完整科学结果。成功产物包括CPU tensor `model.pt`、train_rows.json、validation_rows.json及两集合预测CSV；report.json最后发布，并绑定执行源码、配置、数据身份、全部产物SHA与逐轮选择记录。权重不提交Git。

开跑即建立running.marker；进度写入、临时目录/映射/线程上下文清理全部完成后才原子发布report，最后删除running.marker作为提交点。失败先写failure.json再移除成功report；即使失败标记也无法写出，既有running.marker仍阻止未完成产物加载。失败目录保留，不伪装成功，也不复用同一输出路径。重载校验完整轮数/验证interval/原衰减点、选中epoch、配置摘要、CPU权重及元数据、完整文件表和当前执行源码；先按登记seed确定性设置，再创建模型并移至cpu/cuda。

训练器导入时捕获执行源码字节，训练前和完成时校验文件未变；另将内存中模块自定义函数/类方法的Python代码与当前源码独立编译结果比较，拒绝依赖模块更早导入后被编辑的情况。同时通过AST literal_eval核对位置及关键字默认参数；检查只编译、不执行源码，也不改写旧冻结模块。源码变化后必须使用新进程重新登记，不能将旧内存函数归属到新磁盘文件。

独立核验工具手写身份裁剪、标准化、概率平均和CSV统计，不调用训练器的预测函数；验证脚本/输入/输出身份与实际设备一并记录。CPU/GPU概率可能有数值差异，同设备保存重载的类别必须一致；跨设备差异须实测记录，不将任意近似概率当成损坏。

## 复跑入口

组件/训练器最小验收：

```bash
.venv/bin/python scripts/train_paper_baseline.py --seed 0 --smoke --device cpu --output results/paper_training/new_cpu_smoke
.venv/bin/python scripts/train_paper_baseline.py --seed 0 --smoke --device cuda --output results/paper_training/new_cuda_smoke
```

`--smoke`仅两轮、各集合每类2条，不代表完整方法成绩。完整训练应使用已冻结registry和它生成的每个run config/输出目录；相同配置可通过`--device cpu`或`--device cuda`运行。CUDA须在可访问GPU的上下文执行。公开clone中的权重和原始数据需要按既有数据说明本地准备。

本轮是有限预算方法重建，包含已公开的工程假设，不能称原文10000轮严格复现。既有formal_v1结果保持不变；后续方法选择只用train/validation，已见test不能作为新的未见确认集。

## 本轮执行入口

终版264项回归及CPU/CUDA独立核验通过后，执行registry冻结于results/experiments/paper_baseline_v1/registry.json。训练配置/registry在三个完整结果前绑定，不能由后续分数改写。示例：

```bash
.venv/bin/python scripts/train_paper_baseline.py --config results/experiments/paper_baseline_v1/configs/resnext_s0.json --registry results/experiments/paper_baseline_v1/registry.json --output results/experiments/paper_baseline_v1/resnext_s0 --device cuda
.venv/bin/python scripts/verify_paper_run.py --run results/experiments/paper_baseline_v1/resnext_s0 --output results/experiments/paper_baseline_v1/verification/resnext_s0_cuda.json --device cuda --split both
```

分别使用登记的seed1/2配置与目录完成其余运行。`execution_status.json`记录可变状态，registry/三个配置不变；不得将正在写入的progress.json当最终成绩。
