# Google Colab GPU训练指南

> **最后更新**: 2026年1月15日
> **适用版本**: XGBoost 2.0+

---

## 快速开始

### 一键运行脚本

将以下代码复制到Colab notebook单元格中运行：

```python
# ============================================================
# CFST XGBoost - Colab GPU 训练脚本
# ============================================================

# 1. 克隆项目
!git clone https://github.com/你的用户名/xgboost.git
%cd xgboost

# 2. 安装依赖（GPU版本）
!pip install -q -r requirements.txt
!pip install -q xgboost --upgrade

# 3. 切换到GPU模式
!sed -i -e 's/device: "cpu"/device: "cuda"/' -e 's/n_jobs: -1/n_jobs: 1/' config/config.yaml

# 4. 验证GPU可用
import torch
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# 5. 开始训练
!python train.py --config config/config.yaml

# 6. 保存结果到Google Drive
from google.colab import drive
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
drive.mount('/content/drive')
!cp -r output /content/drive/MyDrive/xgboost_results_{timestamp}
print(f"✅ 结果已保存到: /content/drive/MyDrive/xgboost_results_{timestamp}")
```

---

## 详细说明

### 第一步：获取GPU运行时

1. 打开 Colab: https://colab.research.google.com/
2. 点击 **运行时** → **更改运行时类型**
3. 硬件加速器选择 **T4 GPU**
4. 点击保存

### 第二步：克隆项目

**方式A：使用公共仓库**
```python
!git clone https://github.com/用户名/xgboost.git
%cd xgboost
```

**方式B：使用私有仓库**
```python
# 需要先配置GitHub访问令牌
!git clone https://github.com/用户名/xgboost.git
%cd xgboost
```

### 第三步：安装GPU版本依赖

```python
# 基础依赖
!pip install -q pandas numpy scikit-learn pyyaml joblib matplotlib seaborn optuna

# GPU版本的XGBoost（重要！）
!pip install -q xgboost --upgrade
```

### 第四步：启用GPU模式

代码修改已完成，只需修改配置文件：

```python
# 自动切换到GPU模式
!sed -i -e 's/device: "cpu"/device: "cuda"/' -e 's/n_jobs: -1/n_jobs: 1/' config/config.yaml
```

**验证配置：**
```python
!grep -A 2 "device:" config/config.yaml
```

预期输出：
```yaml
device: "cuda"
n_jobs: 1
```

### 第五步：验证GPU可用

```python
import xgboost as xgb
import torch

print("=" * 50)
print("环境检查")
print("=" * 50)

# PyTorch GPU检查
print(f"✅ PyTorch CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU设备: {torch.cuda.get_device_name(0)}")

# XGBoost GPU检查
try:
    # 测试创建GPU模型
    test_model = xgb.XGBRegressor(device='cuda', tree_method='hist')
    print("✅ XGBoost GPU支持: 正常")
except Exception as e:
    print(f"❌ XGBoost GPU支持: {e}")
```

### 第六步：开始训练

```python
# 基础训练
!python train.py --config config/config.yaml

# 或者自定义输出目录
!python train.py --config config/config.yaml --output output_gpu_run
```

### 第七步：保存结果

**Colab会话结束后文件会丢失，务必保存到Google Drive：**

```python
from google.colab import drive
import datetime

# 挂载Drive
drive.mount('/content/drive')

# 创建带时间戳的备份目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/xgboost_results_{timestamp}"

# 复制结果
!cp -r output {backup_path}
!cp -r logs {backup_path}/logs
!cp -r plots {backup_path}/plots  # 如果有

print(f"✅ 结果已保存到: {backup_path}")
```

---

## 完整Colab Notebook模板

```python
# %% [markdown]
# # CFST XGBoost - GPU训练
# > 生成时间: 2026-01-15

# %% [markdown]
# ## 1. 环境设置

# %%
!git clone https://github.com/你的用户名/xgboost.git
%cd xgboost

# %% [markdown]
# ## 2. 安装依赖

# %%
!pip install -q -r requirements.txt
!pip install -q xgboost --upgrade

# %% [markdown]
# ## 3. GPU验证

# %%
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 4. 启用GPU模式

# %%
!sed -i -e 's/device: "cpu"/device: "cuda"/' -e 's/n_jobs: -1/n_jobs: 1/' config/config.yaml
!grep -A 1 "device:" config/config.yaml

# %% [markdown]
# ## 5. 开始训练

# %%
!python train.py --config config/config.yaml

# %% [markdown]
# ## 6. 查看结果

# %%
import pandas as pd
import json

with open('output/evaluation_report.json', 'r') as f:
    report = json.load(f)

print("=" * 50)
print("训练结果")
print("=" * 50)
print(f"训练集 RMSE: {report['train_metrics_original_space']['rmse']:.4f}")
print(f"测试集 RMSE: {report['test_metrics_original_space']['rmse']:.4f}")
print(f"测试集 R²: {report['test_metrics_original_space']['r2']:.4f}")
print(f"测试集 COV: {report['test_metrics_original_space']['cov']:.4f}")

# %% [markdown]
# ## 7. 保存到Google Drive

# %%
from google.colab import drive
import datetime

drive.mount('/content/drive')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
!cp -r output /content/drive/MyDrive/xgboost_results_{timestamp}
print(f"✅ 已保存到: xgboost_results_{timestamp}")
```

---

## 常见问题

### Q1: 提示CUDA不可用

**解决方案：**
1. 确保选择了GPU运行时（运行时 → 更改运行时类型 → T4 GPU）
2. 重启运行时（运行时 → 重启运行时）

### Q2: XGBoost GPU初始化失败

**错误示例：**
```
XGBoostError: [16:04:39] WARNING: ... GPU support not detected
```

**解决方案：**
```python
# 卸载旧版本，重新安装GPU版本
!pip uninstall xgboost -y
!pip install xgboost --upgrade
```

### Q3: 内存不足

**错误示例：**
```
CUDA out of memory
```

**解决方案：**
```python
# 方法1：减少n_estimators
!sed -i 's/n_estimators: [0-9]*/n_estimators: 400/' config/config.yaml

# 方法2：减少batch size或数据量
# 修改数据加载逻辑，使用部分数据
```

### Q4: 训练中断后如何恢复

```python
# Optuna会自动保存进度到 logs/optuna_study.db
# 直接重新运行即可从断点继续
!python train.py --config config/config.yaml
```

---

## 性能对比

| 环境 | 设备 | 300次Optuna预估时间 |
|------|------|-------------------|
| 本地 | CPU (8核) | ~2-3小时 |
| Colab | T4 GPU | ~30-45分钟 |
| Colab Pro | V100 GPU | ~15-20分钟 |

---

## 配置参数说明

### GPU模式推荐配置

```yaml
model:
  params:
    device: "cuda"          # 使用GPU
    n_jobs: 1              # 避免GPU资源冲突
    tree_method: "hist"    # GPU优化的直方图算法
```

### CPU模式配置

```yaml
model:
  params:
    device: "cpu"          # 使用CPU
    n_jobs: -1             # 使用所有CPU核心
    tree_method: "hist"    # 直方图算法
```

---

## 技术支持

- **项目仓库**: https://github.com/你的用户名/xgboost
- **问题反馈**: GitHub Issues
- **更新日期**: 2026-01-15

---

## 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-01-15 | v1.0 | 初始版本，支持GPU/CPU动态切换 |
