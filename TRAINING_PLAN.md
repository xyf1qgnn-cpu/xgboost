# CFST柱极限承载力预测 - 训练执行计划

## 🎯 项目目标

基于多截面统一CFST柱极限承载力预测，使用去重后数据集，实现**COV < 0.05**的高稳定性预测模型。

---

## 📊 数据集概况

**文件**: `data/raw/feature_parameters_unique.csv`
**样本量**: 4084个（已去重）
**截面类型**: 圆形、矩形、方形、圆端形
**承载力范围**: 100.44 kN - 46000.00 kN

**分布特征**:
- 均值: 2480.58 kN
- 中位数: 1601.50 kN
- 标准差: 3179.38 kN
- **分布**: 严重右偏（均值 > 中位数）
- **离群值**: 380个样本（9.3%）> 5349 kN

**核心挑战**: 离群值会显著增大COV指标，需要强正则化和稳定性措施

---

## 🔍 参数剔除策略

### 剔除的6个参数（基于相关系数矩阵分析）

```yaml
columns_to_drop:
  - "b (mm)"     # r=0.760 with Nexp - 与承载力强线性相关，导致R²虚高
  - "h (mm)"     # r=0.774 with Nexp - 与承载力强线性相关，导致R²虚高
  - "r0 (mm)"    # 信息可由r0/h表示
  - "t (mm)"     # 信息已由As和te表示
  - "L (mm)"     # 信息已由lambda_bar表示
  - "lambda"     # 信息已由lambda_bar表示
```

**剔除理由**:
1. b和h与Nexp存在极强线性关系（0.76-0.77），这是R²虚高的根本原因
2. 模型过度依赖这些直接尺寸特征，而非学习复杂力学关系
3. 强制使用无量纲参数，符合"多截面统一预测"的工程原理

**保留参数**: 18个（材料5 + 几何7 + 约束2 + 长细1 + 偏心5）

### 相关系数矩阵关键发现

```
几何参数相关性：
- b与h: 0.911 (极高相关，冗余信息)
- b与Nexp: 0.760 (强线性，导致R²虚高)
- h与Nexp: 0.774 (强线性，导致R²虚高)

无量纲参数合理性：
- Ac与Nexp: 0.798 (合理，面积相关)
- As与Nexp: 0.872 (合理，钢管直接承载)
- lambda_bar与Nexp: -0.143 (合理，长细比影响稳定)
- e/h与Nexp: -0.209 (合理，偏心率降低承载力)
```

---

## ⚙️ 两阶段训练策略

### Stage 1: 基础训练（评估COV基准）

**配置文件**: `config/config.yaml`
**启用Optuna**: `use_optuna: false`
**目标**: 评估参数剔除后，COV改善情况

**核心参数配置**:
```yaml
model:
  params:
    max_depth: 4              # ↓ 从6降至4，防止过拟合
    learning_rate: 0.08       # ↓ 从0.1降至0.08，稳定收敛
    n_estimators: 400         # ↑ 从200增至400，补偿深度降低
    subsample: 0.9            # ↑ 从0.8增至0.9，提高稳定性
    colsample_bytree: 0.9     # ↑ 从0.8增至0.9，提高稳定性
    min_child_weight: 8       # ↑↑ 从1增至8，应对9.3%离群值
    reg_alpha: 0.5            # + 新增L1正则化
    reg_lambda: 2.0           # + 新增L2正则化
    gamma: 0.1                # + 最小损失减少
```

**调参理由**:
- **max_depth↓**: 减少过拟合，防止模型学习离群值特征
- **min_child_weight↑↑**: 关键！380个离群值要求每个叶子节点有足够样本
- **n_estimators↑**: 更多棵树补偿单棵树深度降低
- **正则化**: 强制提高泛化能力，降低对离群值敏感度

**执行命令**:
```bash
# Stage 1: Evaluate baseline COV
python train.py --config config/config.yaml

# 预期结果：
# - R²: 可能从虚高0.99+降至0.95-0.97（更真实）
# - COV: 目标<0.15，若>0.15则进入Stage 2
# - 训练时间: ~30-60秒
```

**关键输出关注点**:
1. **COV值**:
   - 如果COV < 0.10: ✅ 优秀，考虑结束训练
   - 如果COV < 0.15: ✅ 良好，根据R²决定是否调优
   - 如果COV > 0.15: ⚠️ 进入Stage 2（可能需要启用Optuna）

2. **R²值**:
   - 可能下降0.02-0.05（这是正常的，因为剔除了线性特征）
   - 重点关注R²下降幅度vs COV改善幅度

---

### Stage 2: Optuna深度调优（目标COV < 0.05）

**配置文件**: `config/config.yaml`
**修改步骤**: 在config.yaml中设置 `use_optuna: true`
**目标**: 通过超参数搜索，实现COV < 0.05

**Optuna配置**（在config.yaml中）:
```yaml
model:
  use_optuna: true           # 从 false 改为 true
  n_trials: 200              # 200次试验，更充分搜索
  optuna_timeout: 7200       # 2小时超时
```

**搜索空间优化**（已在src/model_trainer.py中配置）:
```python
# 为应对离群值和COV<0.05目标，搜索空间已优化：
max_depth:     trial.suggest_int(3, 7)        # 上限从10降至7
learning_rate: trial.suggest_float(0.05, 0.2, log=True)
n_estimators:  trial.suggest_int(250, 600)
subsample:     trial.suggest_float(0.85, 0.98)
colsample:     trial.suggest_float(0.85, 0.98)
min_child_weight: trial.suggest_int(5, 20)    # 关键！从5-20，应对离群值
reg_alpha:     trial.suggest_float(0.1, 2.0, log=True)
reg_lambda:    trial.suggest_float(1.0, 10.0, log=True)
gamma:         trial.suggest_float(0.05, 0.3)
```

**执行步骤**:
```bash
# 1. 编辑 config/config.yaml
#    将 use_optuna: false 改为 use_optuna: true

# 2. 执行Stage 2训练
python train.py --config config/config.yaml

# 预期结果：
# - 训练时间: ~30-60分钟（200次试验）
# - COV目标: <0.05（优秀）
# - R²目标: 保持>0.95

# 3. 训练完成后恢复配置（可选）
#    将 use_optuna: true 改回 use_optuna: false
```

**搜索策略**:
- Optuna使用TPE（Tree-structured Parzen Estimator）算法
- 每次试验使用5折交叉验证评估RMSE
- 记录最佳参数组合和对应的COV、R²

**如果Stage 2后COV仍>0.05**:
考虑以下补充措施（任选其一）：
1. 增加n_trials到300，扩展搜索空间
2. 在model_trainer.py中进一步收紧正则化参数范围
3. 检查离群值样本的预测误差分布，针对性改进
4. 考虑特征工程（如增加交互项）

---

## 📋 执行步骤

### Step 1: 执行Stage 1训练

```bash
cd /home/thelya/Work/xgboost

# Run Stage 1 (Baseline)
python train.py --config config/config.yaml
```

**预期输出**:
```
================================================================================
CFST XGBOOST PIPELINE - TRAINING STARTED
================================================================================
...
Step 6: Evaluating model...
Model evaluation completed:
  RMSE: 250.45
  MAE: 145.32
  R²: 0.9612          # 可能从0.99+降至0.96左右
  MAPE: 8.23%
  COV: 0.1256 (μ≈1.0 indicates no bias, <0.10 excellent)  # 重点关注！
...
================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================
```

**决策点**:
- 如果COV < 0.10: ✅ 优秀，记录参数，可选择结束
- 如果COV < 0.15: ✅ 良好，根据实际情况决定是否进入Stage 2
- 如果COV > 0.15: ⚠️ 进入Stage 2

### Step 2: 评估Stage 1结果

检查输出文件：
```bash
# 查看评估报告
cat output/evaluation_report.json | grep -A 10 "metrics"

# 查看训练元数据
cat output/training_metadata.json
```

重点关注：
- COV值是否达到预期
- R²下降幅度是否在可接受范围
- 特征重要性是否合理（As、Ac、fy、fc应最重要）

### Step 3: （如果需要）执行Stage 2

```bash
# 1. 编辑 config/config.yaml
#    将 use_optuna: false 改为 use_optuna: true

# 2. 执行Stage 2训练
python train.py --config config/config.yaml

# 监控训练进度（另开一个终端）
tail -f output/ml_pipeline.log
```

**监控指标**:
```bash
# 每完成一个trial，查看中间结果
# 关注 Best RMSE 和对应的参数
```

### Step 4: 对比Stage 1和Stage 2结果

```bash
# Stage 1结果
echo "Stage 1 - COV: $(cat output/evaluation_report.json | jq '.metrics.cov')"
echo "Stage 1 - R²: $(cat output/evaluation_report.json | jq '.metrics.r2')"

# Stage 2结果（Stage 2会覆盖 output/ 目录）
echo "Stage 2 - COV: $(cat output/evaluation_report.json | jq '.metrics.cov')"
echo "Stage 2 - R²: $(cat output/evaluation_report.json | jq '.metrics.r2')"
```

### Step 5: 选择最优模型

**决策标准**（按优先级排序）：
1. **COV < 0.05**: 是否达标？
2. **R² > 0.95**: 是否可接受下降幅度？
3. **训练时间**: 是否在实际应用可接受范围？
4. **特征重要性**: 是否合理符合工程原理？

**预期最终结果**:
- **COV**: 0.04-0.08（优秀<0.1）
- **R²**: 0.94-0.97（比Stage 1略有提升）
- **训练时间**: 30-60分钟（Stage 2）

---

## 📊 成功标准

### COV达标等级

| COV范围 | 等级 | 结构工程标准 | 决策 |
|---------|------|--------------|------|
| < 0.05  | 🏆 极好 | 优秀预测稳定性 | ✅ 达标，记录参数 |
| 0.05-0.07 | 🥇 优秀 | 生产部署推荐 | ✅ 达标，推荐部署 |
| 0.07-0.10 | 🥈 良好 | 可接受 | ✅ 达标，可考虑部署 |
| 0.10-0.15 | 🥉 一般 | 需增加安全裕度 | ⚠️ 可接受，建议继续优化 |
| > 0.15 | ❌ 不佳 | 不建议部署 | 🔄 继续调优 |

**用户要求**: COV < 0.05（优秀等级）

### R²合理性范围

| R²范围 | 说明 | 备注 |
|--------|------|------|
| 0.98-1.00 | 😐 可能虚高 | 检查是否依赖线性特征 |
| 0.95-0.98 | ✅ 优秀 | 预期范围 |
| 0.90-0.95 | ✅ 良好 | 可接受 |
| < 0.90 | ⚠️ 需改进 | 考虑特征工程或更多数据 |

**预期**: 参数剔除后R²可能下降0.02-0.05，这是正常的工程合理性调整

### 综合评估矩阵

| COV \ R² | >0.98 | 0.95-0.98 | 0.90-0.95 | <0.90 |
|----------|-------|-----------|-----------|-------|
| **<0.05** | ✅🏆 | ✅🏆 | ✅🥈 | ⚠️ |
| **0.05-0.07** | ✅🏆 | ✅🥇 | ✅🥈 | ⚠️ |
| **0.07-0.10** | ✅🥇 | ✅🥈 | ⚠️ | ❌ |
| **>0.10** | ⚠️ | ⚠️ | ❌ | ❌ |

**理想目标**: COV < 0.05 + R² > 0.95（绿色区域）

---

## 🐛 问题排查指南

### 问题1: Stage 1后COV仍>0.15

**可能原因**:
1. 离群值过多（确认是否超过10%）
2. 正则化不够强
3. 特征选择不当

**解决方案**:
直接进入Stage 2，Optuna会自动找到更好的正则化组合

### 问题2: Stage 2训练时间过长

**可能原因**:
1. n_trials设置过高
2. 数据量过大
3. 交叉验证计算开销

**解决方案**:
```yaml
# 在config/config.yaml中调整：
model:
  n_trials: 150        # 从200降低到150
  optuna_timeout: 3600 # 从7200降低到3600秒
```

### 问题3: Stage 2后COV改善不明显

**可能原因**:
1. 搜索空间仍不够大
2. 离群值影响过大
3. 需要目标变量变换

**解决方案**:
1. 在model_trainer.py中进一步扩大正则化范围：
```python
min_child_weight: trial.suggest_int(10, 30)  # 从5-20提高到10-30
reg_lambda: trial.suggest_float(5.0, 20.0)   # 从1-10提高到5-20
```

2. 考虑对目标变量进行对数变换（可选）：
```python
# 在src/preprocessor.py中添加
from sklearn.preprocessing import FunctionTransformer

# Option: Log transform target to handle outliers
# Note: Requires inverse_transform in predictor.py
```

### 问题4: R²下降过多（<0.90）

**可能原因**:
1. 特征剔除过多
2. 正则化过强
3. 模型复杂度不足

**解决方案**:
1. 检查是否可以恢复1-2个重要特征
2. 降低min_child_weightslightly
3. 增加max_depth slightly
4. 启用特征选择，找到最优子集

---

## 📈 预期效果

### 对比：之前配置 vs Stage 1配置

| 参数 | 之前配置 | Stage 1 | 变化 | 理由 |
|------|---------|---------|------|------|
| max_depth | 6 | 4 | -33% | 防止过拟合 |
| n_estimators | 200 | 400 | +100% | 补偿深度降低 |
| min_child_weight | 1 | 8 | +700% | **关键！**应对9.3%离群值 |
| subsample | 0.8 | 0.9 | +12.5% | 提高稳定性 |
| reg_alpha | 0 | 0.5 | +∞ | L1正则化 |
| reg_lambda | 1 | 2 | +100% | L2正则化增强 |

### 预期指标改善

| 指标 | 之前结果 | Stage 1预期 | Stage 2目标 | 说明 |
|------|---------|------------|------------|------|
| **R²** | 0.995（虚高） | 0.95-0.97 | 0.96-0.98 | 可能略降，但更健康 |
| **COV** | 0.116（不佳） | 0.08-0.12 | **<0.05** | **主要目标** |
| **RMSE** | 226 | 240-280 | 230-260 | 略有增加 |
| **MAPE** | 7.65% | 8-10% | 7-9% | 基本保持 |

---

## 📝 记录和报告

### 需要记录的内容

每次训练后，记录以下信息：

1. **配置信息**:
   - Config文件版本
   - 关键参数取值
   - 训练时间

2. **性能指标**:
   - R², RMSE, MAE, MAPE
   - **COV值**（最重要）
   - 训练/测试集分布

3. **工程合理性**:
   - 特征重要性排名
   - As, Ac, fy, fc是否排名靠前（预期）
   - 预测误差分布

4. **对比分析**:
   - 剔除参数前后的R²变化
   - COV改善幅度
   - 是否达到COV < 0.05目标

### 报告模板

**训练实验报告 - Stage X**

```markdown
## 实验配置
- 数据集: feature_parameters_unique.csv (4084 samples)
- 剔除参数: b, h, r0, t, L, lambda
- 保留参数: 18个
- 配置版本: config/config.yaml (Stage 1)

## 性能结果
- R²: 0.XX
- COV: 0.XX (<0.05? ✓/✗)
- RMSE: XXX kN
- MAPE: X.X%

## 关键发现
1. [发现1]
2. [发现2]

## 下一步行动
- [ ] 进入Stage 2（如果COV > 0.05）
- [ ] 部署模型（如果COV < 0.05）
- [ ] 调整参数（如果R²过低）
```

---

## ✅ 任务清单

### 立即执行任务

- [x] **分析相关系数矩阵** - 确定剔除6个参数
- [x] **配置训练文件** - `config/config.yaml` (通过use_optuna切换阶段)
- [x] **优化Optuna搜索空间** - `src/model_trainer.py`
- [x] **创建训练计划文档** - `TRAINING_PLAN.md`
- [x] **简化配置管理** - 删除冗余配置文件，仅保留config.yaml

### 待执行任务

- [ ] **执行Stage 1训练** - 评估基准COV (use_optuna: false)
- [ ] **验证COV指标** - 判断是否进入Stage 2
- [ ] **（如果需要）执行Stage 2** - 修改use_optuna: true，启用Optuna调优
- [ ] **验证最终指标** - 确保COV < 0.05
- [ ] **模型部署准备** - 保存最优模型

---

## 🎯 下一步行动

请执行以下命令开始训练：

```bash
# 1. 确认数据集路径
cat config/config.yaml | grep file_path
# 应显示: data/raw/feature_parameters_unique.csv

# 2. 执行Stage 1训练
cd /home/thelya/Work/xgboost
python train.py --config config/config.yaml

# 3. 观察输出，重点关注COV值
# 训练完成后，查看结果：
cat output/evaluation_report.json
```

**训练完成后**，请告诉我：
1. **COV值是多少？**（是否<0.05或<0.10？）
2. **R²值是多少？**（下降幅度如何？）
3. **特征重要性排名是否合理？**

根据这些结果，我们再决定是否需要进入**Stage 2**（启用Optuna）！

---

**文档版本**: v1.0
**创建日期**: 2026-01-14
**最后更新**: 2026-01-14
