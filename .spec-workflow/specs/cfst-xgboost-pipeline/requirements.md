# Requirements Document

## Introduction

构建一个基于XGBoost的机器学习管道，用于预测CFST（钢管混凝土）柱的极限承载力。该系统将利用经过特征工程处理的数据（feature_parameters.csv），通过剔除绝对几何尺寸参数（b, h, r₀, t, L, λ），构建一个多截面统一的预测模型。

## Alignment with Product Vision

本项目旨在为结构工程领域提供一个准确、可靠的CFST柱承载力预测工具，通过机器学习方法提高设计效率，减少实验成本，并为工程实践提供决策支持。

## Requirements

### Requirement 1: 数据加载与预处理

**User Story:** 作为结构工程师，我希望系统能正确加载CFST柱的特征数据和标签数据，并自动剔除指定的几何参数，以便构建统一的预测模型。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN 系统 SHALL 从feature_parameters.csv加载数据
2. IF 数据文件存在且格式正确 THEN 系统 SHALL 识别K列为标签列
3. WHEN 数据加载完成 THEN 系统 SHALL 自动剔除列b, h, r₀, t, L, λ
4. IF 剔除后数据维度正确 THEN 系统 SHALL 将剩余特征作为模型输入

### Requirement 2: XGBoost模型训练

**User Story:** 作为机器学习工程师，我希望系统能实现XGBoost回归模型的训练，包含超参数调优和交叉验证，以获得最佳的预测性能。

#### Acceptance Criteria

1. WHEN 训练数据准备完成 THEN 系统 SHALL 将数据分割为训练集和测试集（默认80/20）
2. IF 训练集数量充足 THEN 系统 SHALL 使用5折交叉验证进行模型训练
3. WHEN 训练过程中 THEN 系统 SHALL 记录RMSE、MAE、R²等性能指标
4. IF 启用超参数优化 THEN 系统 SHALL 使用Optuna进行超参数搜索
5. WHEN 模型训练完成 THEN 系统 SHALL 保存最优模型和训练日志

### Requirement 3: 模型评估与验证

**User Story:** 作为项目负责人，我希望系统能全面评估模型性能，包括预测准确性、泛化能力和特征重要性分析，以确保模型的可靠性。

#### Acceptance Criteria

1. WHEN 模型训练完成后 THEN 系统 SHALL 在测试集上进行预测
2. IF 预测完成 THEN 系统 SHALL 计算并显示RMSE、MAE、R²、MAPE等指标
3. WHEN 评估指标计算完成 THEN 系统 SHALL 生成预测值vs真实值散点图
4. IF 用户请求 THEN 系统 SHALL 提供特征重要性排名和可视化
5. WHEN 评估完成 THEN 系统 SHALL 生成详细的评估报告

### Requirement 4: 模型预测功能

**User Story:** 作为终端用户，我希望系统能提供加载已训练模型并对新数据进行预测的功能，支持批量预测和结果导出。

#### Acceptance Criteria

1. WHEN 用户加载训练好的模型 THEN 系统 SHALL 验证模型文件的完整性
2. IF 输入数据格式正确 THEN 系统 SHALL 使用模型进行预测
3. WHEN 预测完成 THEN 系统 SHALL 返回预测承载力值
4. IF 批量预测 THEN 系统 SHALL 处理多条记录并返回结果列表
5. WHEN 预测结果生成 THEN 系统 SHALL 支持导出为CSV格式

### Requirement 5: 模型持久化与管理

**User Story:** 作为系统管理员，我希望系统能保存训练好的模型和相关元数据，支持模型版本管理和快速加载。

#### Acceptance Criteria

1. WHEN 模型训练完成 THEN 系统 SHALL 保存模型文件（.pkl格式）
2. IF 保存成功 THEN 系统 SHALL 同时保存特征列表和参数配置
3. WHEN 用户请求加载模型 THEN 系统 SHALL 从指定路径加载模型
4. IF 模型文件损坏或不匹配 THEN 系统 SHALL 抛出清晰的错误信息
5. WHEN 模型更新时 THEN 系统 SHALL 支持版本控制和回滚

### Requirement 6: 数据可视化

**User Story:** 作为研究人员，我希望系统能生成特征重要性分析图，帮助我理解模型的关键特征。

#### Acceptance Criteria

1. WHEN 模型训练完成后 THEN 系统 SHALL 生成XGBoost特征重要性图
2. IF 特征重要性计算完成 THEN 系统 SHALL 显示特征重要性排名
3. WHEN 可视化生成后 THEN 系统 SHALL 保存图表文件

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: 数据加载、预处理、训练、评估、预测功能应分离
- **Modular Design**: 各模块应独立可测试，支持复用
- **Clear Interfaces**: 明确定义数据流和API接口
- **Configuration Management**: 使用配置文件管理超参数和数据路径

### Performance
- 模型训练应支持并行计算，利用多核CPU
- 单次预测响应时间应小于100ms
- 批量预测应支持至少10000条记录的高效处理

### Reliability
- 数据加载和预处理过程应有完整性检查
- 模型训练过程应有异常处理和恢复机制
- 预测过程应对输入数据进行验证和清洗
- 系统应提供详细的日志记录，便于故障排查
