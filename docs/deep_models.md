# CNN、ResNet与附加噪声

本阶段提供开发训练与最小验收链路，测试分类指标继续封存。实施计划见[deep-noise](plans/2026-10-01-deep-noise.md)，协议变更见[D011](decisions.md)。类别语义尚未确认，只输出class_index指标。

## 安装与设备

Python3.12；先安装已有数据/SVM依赖，再选一种官方PyTorch2.9.1构建。

```bash
uv pip install --python .venv/bin/python -r requirements-models.txt
# 仅CPU环境
uv pip install --python .venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA CUDA环境：与CPU命令二选一
uv pip install --python .venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-deep.txt
```

CPU/CUDA构建安装命令来自[PyTorch官方版本页](https://pytorch.org/get-started/previous-versions/)。不需要torchvision/torchaudio，不在项目内安装或修改系统驱动。此环境安装2.9.1+cu128，但CUDA不可用；CPU已验证，真实GPU尚未验收。

`--device auto`按torch.cuda.is_available选择；`cpu`明确使用CPU；`cuda`不可用时明确报错，不静默回退。安装了CUDA包并不等于硬件路径可用。CPU可以训练、计算指标和加载权重；不做卡型号分支或CPU/GPU预算评估。

## 网络与训练

输入共用250×2原单位缓存，固定训练mean/std后转[B,2,250] float32，保留输入直流，不裁剪。CNN三块32/64/128通道，Conv(kernel5)-BN-ReLU-MaxPool2，GAP后分类。ResNet stem32(kernel7)，32/64/128各两个basic block(kernel3)，阶段首块stride1/2/2，GAP后分类。二者规模不同，不把表现差异单独归因于残差连接。

config/deep_v1.json默认AdamW、交叉熵、lr0.001、weight_decay0.0001、batch128、最多30epoch、patience8。普通CNN/ResNet按clean验证Macro-F1选epoch；增强ResNet按Clean/30/20/10dB等权Macro-F1，均按Accuracy/较早epoch打破tie。每epoch验证噪声固定seed10和namespace=validation；增强与普通模型选择目标不同，结果解释应注明。

训练seed控制初始化、shuffle与增强；sampling_seed固定20261001使最小验收各模型使用相同样本。启用PyTorch确定性模式，禁用TF32和cuDNN benchmark，记录真实软件和设备；不同设备/版本仍可能产生数值差异。[PyTorch随机性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)明确跨平台/版本的复现限制。

## 噪声口径

src/ect/noise.py不依赖Torch。先用干净原单位缓存计算逐通道去均值后的联合AC功率，再生成两通道同一sigma的高斯向量，去除各通道噪声时间均值并按联合能量校准，最后加回原始信号。它是有限长度能量校准高斯扰动，中心化/能量校准引入相关性，不称为严格独立白噪声。

返回float64带噪原单位信号，随后以固定训练统计量标准化为float32；避免先写回float32原单位使微小扰动被直流舍入。info记录干净AC功率和实际返回残差功率，零AC保持输入、snr_applicable=false。验证报告记录实际SNR偏差和不适用计数，预测CSV也标记N/A。

SHA256编码(namespace,seed,wave_sha256,条件)生成PCG64种子，同一身份/条件跨模型、设备、批大小和顺序保持相同扰动，不用Python hash。训练namespace=training，选择键还含epoch：50%保持干净、50%均匀10–30dB；伯努利比例不要求每批恰好一半。验证seed10，正式验证重复10–14、测试100–104留在实验阶段；本次没有调用test分类。

此噪声是在解调并降采样信号上加入的相对扰动，不代表仪器噪声底或工业现场干扰。

## 运行与重载

先准备data/processed/grouped_v1缓存，见[data_pipeline](data_pipeline.md)。三个最小验收命令：

```bash
.venv/bin/python scripts/train_deep.py --model cnn --device cpu --smoke --output results/deep/cnn_smoke_v1
.venv/bin/python scripts/train_deep.py --model resnet --device cpu --smoke --output results/deep/resnet_smoke_v1
.venv/bin/python scripts/train_deep.py --model resnet --augment --device cpu --smoke --output results/deep/resnet_aug_smoke_v1
```

`--smoke`只用train/validation每类10条、两epoch；只作实现验收，不同全量SVM分数比较。去掉--smoke即使用完整train/validation默认开发配置；训练种子可用--seed，设备可换auto/cuda。正式多种子比较另行登记，不能用封存test来修改配置。

输出目录原子占用，拒绝覆盖；复跑选新--output。report最后发布，失败保存failure.json并清理成功标记/权重。记录配置、代码、原输入/清单/缓存SHA、选中epoch、各条件/组指标、逐epoch损失与增强数量、预测及抽样行索引。读取器完整性检查仍哈希并映射三集合缓存文件，但训练、统计评价和分类仅访问train/validation；这里封存的是test分类/选参，不是禁止test文件身份检查。

model.pt包含CPU state_dict、结构和冻结标准化，不提交Git；普通克隆后复跑生成权重。用[PyTorch推荐的state_dict与map_location](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)做可移植加载，weights_only=True；先检查产物、配置摘要、选中epoch/指标记录与LF规范化源码SHA，再核对checkpoint元数据并加载模型。同训练设备的重载预测须一致；CPU迁移另记logits最大误差、rtol=1e-4/atol=1e-5范围判断、argmax差异数量和CPU指标。跨设备微小数值差导致边界类别变化不删除有效权重；本环境只有CPU真实验收，GPU→CPU差异尚无实测。

```python
from src.ect.deep import load_deep_model, predict_raw
bundle = load_deep_model('results/deep/cnn_smoke_v1', device='cpu')
# raw为原单位[N,250,2]，ids为对应manifest的wave_sha256
prediction = predict_raw(bundle, raw, ids, snr_db=20, seed=10, namespace='validation')
```

测试命令：`.venv/bin/python -m unittest discover -s tests -v`。CUDA真实梯度/CPU重载测试按可用性执行；跳过不计为GPU验收。原生Windows尚未运行验收。
