# 数据完整性阶段实施计划

**目标：** 将临时数据观察变为可复跑的完整审计，并给出具备证据的划分约束。

**设计：** [研究设计](../research_design.md)。Python/NumPy mmap 分人员统计数值，逐条 SHA256 建立样本表和人员连接图；不依赖未知的标签映射。

**约束：** 不改变原始文件；人员标识含文件名防止 train/test索引冲突；统计含非有限、常量、跨标签、跨人员和跨集合重复；时间序列哈希包含 shape/dtype。结果自带文件SHA256和版本。

## T1 可复跑审计

文件：`src/ect/audit.py`、`scripts/audit_data.py`、`tests/test_audit.py`、`requirements-audit.txt`。

- [x] 先在小型七维数据构造跨集合重复、跨人员连接的传递链、跨标签重复与非有限值；测试预期统计和连接分量。
- [x] 运行 unittest，确认因为审计函数缺失而失败。
- [x] 实现 `audit_arrays(arrays: dict[str, np.ndarray]) -> (dict, list[dict])`，返回汇总和逐样本索引/哈希。
- [x] 添加命令行：`python scripts/audit_data.py --output results/data_audit`；默认检查data/raw/两个NPY；mmap读取，JSON/CSV输出。
- [x] 运行测试，再在真实数据完整运行，检查2928条交叉波形与上次结果一致。

## T2 来源与类别映射

输出 `results/provenance/` 保存成功的公开元数据，`findings.md` 保存访问失败和证据边界。

- [x] 核对官方Kaggle文件清单、版本、说明；下载可信文件比对完整SHA256，两文件均一致。
- [ ] 找到索引至深度/特殊类的明确语义说明；未找到则保留未知状态。
- [x] 对人员连接图和去重后类别计数提出可行划分；冻结前检查跨集合完整波形零交集。

## T3 划分清单

文件：`src/ect/splits.py`、`scripts/build_splits.py`、`tests/test_splits.py`，产出 `manifests/grouped_v1/`。

- [x] 测试整连通分量分配，拒绝相同波形跨集合或跨标签，保留重复来源索引；相同种子稳定输出。
- [x] 构建当前v1候选协议：原test:0加两个随机单人分量为测试，另两个单人分量为验证，剩余单人和两个大分量为训练；固定种子20261001。
- [x] 各集合内部去重；记录所有原始来源；绑定输入SHA256及审计清单SHA256。
- [x] 验证24000/3200/4800唯一波形、每类1200/160/240，且哈希和人员组无跨集合交叉。
- [x] 文件身份重算核对后才产出清单。协议只根据审计确定，不依据模型测试表现。

## T4 验证与记录

- [x] 小型测试通过且真实数据审计成功。
- [x] README 给出现有命令；progress写入实际命令/结果。
- [x] task_plan 更新已完成条目和下一步；未知不写成已解决。
