# 近重复诊断与SVM基线实施计划

状态：执行中。用户已授权继续长期项目；本轮范围为诊断和CPU SVM开发验证，不计算最终测试分类性能。

## 固定设计

使用已冻结grouped_v1缓存。两条独立模块共用PreparedDataset，不修改缓存、清单或已有预处理。正式模型比較尚未启动；近重复只能识别指定变换下的候选，不能证明实际采集来源独立。

诊断首先查询全部validation对全部train（3200×24000），不使用类别选择邻居。两种距离：

- 原单位相对误差：`||q-r||₂ / ||q-mean_t(q)||₂`，阈值0.001。非零AC查询以实际差值重算距离，避免大直流带来的数值消减。
- 形状距离：每通道去时间均值后，两个通道联合单位L2归一化；`||u(q)-u(r)||₂`，阈值0.01。不做时间平移、符号翻转、单独通道缩放；零AC不适用。

分块穷举，不建立全量距离矩阵；以float64、稳定去均值及候选直接重算保证精度，索引并列时选最小。记录最近邻、距离、原始SHA/来源、类别是否一致、分位数、候选数、耗时和代码/缓存SHA；类别只用于候选产生后的诊断。形状相近不自动删样本、不改变划分。测试集后续仅按相同冻结阈值审计，不用于调参。

SVM特征从原单位缓存提取，不先做训练通道z分数：每通道10项（均值、总体标准差、峰峰值、10/50/90分位数、平均绝对AC幅值、一阶差分RMS、偏度、超额峰度），加I/Q AC协方差、相关系数、轨迹总长度、首末距离、闭合轨迹有向面积，共25项。常量通道偏度/峰度/相关系数取0，保留均值；不将这些特征解释为已校准物理量。

每个模型为StandardScaler→分类器，Scaler只拟合当前训练样本。LinearSVC C=0.1/1/10，dual=False，tol=1e-4，max_iter=10000；RBF SVC C=1/10，gamma=scale/0.01/0.1，tol=1e-3，max_iter=-1，cache_size=512MB，不拟合概率。九个候选顺序固定；以验证Macro-F1、Accuracy、较早候选顺序选优。未收敛候选不得选优，警告写入记录。

先用seed=20261001在训练每类抽100条做2000条pilot，验证使用全部3200。pilot仅估计预算/验证链路，不能与全量模型作为正式同条件比较；保留样本索引。全量训练按实测预算推进，九项候选/验证规则不因验证表现增加搜索。单进程，默认计算线程1，记录实际线程库与版本。CPU SVM无需CUDA，后续深度模型另实现设备切换。

运行输出新目录（拒绝覆盖），成功后发布元数据、所有候选指标、被选模型、逐条验证预测、混淆矩阵、每类召回、训练索引和模型SHA；失败保存failure.json。model.joblib不上传Git。运行绑定缓存metadata、manifest和特征/训练代码SHA、配置、依赖、计时。加载模型核验SHA并检验重载预测一致。

实现依据：[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)、[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)、[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)、[距离数值边界](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)。

## 执行步骤与验收

- [x] T1：src/ect/similarity.py、scripts/diagnose_similarity.py、tests/test_similarity.py；小夹具验证已知近重复、offset/scale形状不变、零AC、大直流小差值、分块一致及错误输入。先失败后实现；执行完整train/validation诊断，保留数值和候选。
- [x] T2：src/ect/features.py、tests/test_features.py；验证统计/几何已知值、常量有限、offset/scale行为、输入不变/批次一致。先失败后实现25维接口。
- [x] T3：src/ect/baseline.py、scripts/train_svm.py、tests/test_baseline.py、config/svm_v1.json、requirements-models.txt；测试训练拟合Scaler、索引抽样可复现、只访问训练/验证、重载一致、既有目录拒绝、失败记录。先失败后实现训练→选择→保存完整链路。
- [x] T4：执行pilot、评估预算并执行可行的全量搜索；检查重载、运行身份及无测试指标。独立代码审查，修复实质问题并重跑受影响产物。
- [ ] T5：更新三个根记录、设计/决策、README和诊断/基线报告；验收后合入main并推送授权仓库。

下一单元：按诊断结果明确正式协议边界；CPU/GPU可切换CNN/ResNet和噪声函数，多种子正式测试另行冻结。

验收：59项测试通过；两部分独立审查完成并修复实质问题；最终pilot/full九项无警告、均收敛且模型重载一致。实际结果见docs/svm_results.md。发布检查点见progress.md。
