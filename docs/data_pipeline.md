# 数据加载与预处理协议

日期：2026-10-01。协议版本：grouped_v1 + FIR101/line。

## 运行方式

先将官方v1两份NPY放入data/raw/，安装数据依赖，然后运行：

```bash
uv pip install --python .venv/bin/python -r requirements-data.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/diagnose_data.py
```

prepare不覆盖已有缓存；复跑比较需用`--output data/processed/<新名称>`及`--report results/preprocessing/<新名称>.json`。diagnose通过`--cache`指定对应缓存，`--output`指定诊断目录。两个脚本的默认project-root由脚本位置确定，可从其他当前目录运行；显式--project-root也可指定完整项目路径。

## 数据接口

`ManifestDataset(project_root, manifest_dir)`核对原始文件与三个清单校验和，检查来源索引覆盖、类别一致及整组隔离。`iter_batches(split,batch_size)`重新核验代表波形，返回`[B,1250,2]`和`[B]`类别索引。数组是可写副本，原始mmap只读。

`PreparedDataset(project_root,cache_dir)`核对缓存、标准化参数、原文件及冻结清单的身份。`iter_batches(split,batch_size,normalize=True)`返回`[B,250,2]`与int64标签；normalize=False返回原单位下采样缓存。标准化由固定训练mean/std完成。CNN/ResNet调用方转换为`[B,2,250]`，不改变类别索引顺序。

两种读取器都支持with与close()；调用者应及时关闭映射，特别是在Windows移动文件或删除临时目录前。构造/缓存写入中途失败会关闭已经创建的映射。

## 滤波与标准化

- 原采样2500Hz，目标500Hz；5倍降采样，输出250点，时间步0.002秒。
- 显式101系数FIR，Kaiser beta=5，firwin cutoff=250Hz；cutoff是半幅点，不是严格通带/阻带边界。
- resample_poly(1,5)，时间轴-2，line边界延拓。原信号大直流偏置，避免默认零填充制造边缘跳变。
- 内部float64滤波，缓存float32；训练统计量按实际float32缓存拟合，ddof=0，每通道6000000点。
- 缓存保留原始单位和直流；没有逐样本幅值缩放/去均值。统计量稳定合并批次的中心二阶矩，零方差拒绝。
- 后续噪声加入原单位缓存后，再标准化。展示AC波形时去均值仅用于图形显示。

实现依据：[SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)与[firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)。本处理为课程项目固定协议，不宣称与原论文未公布的滤波实现相同。

## 缓存与追溯

data/processed/grouped_v1/包含三组x/y NPY、normalization.json、metadata.json，不上传Git。
metadata记录原文件/清单/缓存校验和、全部FIR系数、依赖和处理代码校验、batch-size及耗时。先写临时目录，完整成功后才发布；既有缓存不覆盖。

results/preprocessing/保留运行metadata、normalization、verification和训练数据图，上传仓库供审阅。prepare_run.json为生成时快照，后续代码变更若影响处理需重新运行并更新记录。

## 本次全量验证

- train/validation/test为24000/3200/4800，float32 `[N,250,2]`，标签int64，全部有限，20类计数1200/160/240。
- 下采样后32000波形仍各不相同，三集合交叉完整波形数0。
- 标准化训练集均值约`[-3.4e-11,-1.8e-11]`，标准差约`[1,1]`。
- 当前CPU审查修复后生成约1.63秒，缓存约62MiB；这是当前环境实测，不是其他CPU/GPU训练耗时估计。
- 训练波形总功率/AC功率中位比约81.12dB，无零AC功率样本。按总功率生成相对噪声会受到直流偏置主导。
- 图像仅使用训练数据，类名只标index；视觉相似性不构成类别语义或独立来源证据。
- 32项测试通过，包括校验表缺项拒绝、部分映射创建/读取失败关闭、恒定float64方差拒绝。当前环境验证了句柄关闭行为，未进行原生Windows或GPU训练验收。

## 尚未解决

NPY索引的深度/特殊类别语义没有直接证据；近重复和实际采集来源独立性未确认。完整波形无交叉只解决已知直接重复，后续需要明确诊断近重复，不能将此协议夸大为所有形式的泄漏都已排除。
