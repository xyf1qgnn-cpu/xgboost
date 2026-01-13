# Tasks Document - CFST XGBoost Pipeline

## Task 1: 创建项目基础结构和配置文件

- [x] 1.1. 创建项目目录结构
  - 创建 `src/` 目录及子目录
  - 创建 `data/`、`output/`、`models/` 目录
  - 创建 `__init__.py` 文件
  - _Prompt: Role: Python Developer | Task: Create project directory structure and initialization files for CFST XGBoost pipeline, following the design specified in .spec-workflow/specs/cfst-xgboost-pipeline/design.md | Restrictions: Follow Python package conventions, create proper __init__.py files, maintain clean directory organization | Success: All directories created with proper structure, __init__.py files present, project is importable

- [x] 1.2. 创建配置文件 config.yaml
  - File: `config/config.yaml`
  - 配置数据路径、模型参数、训练设置
  - 包含数据配置、XGBoost参数、交叉验证、路径配置
  - _Prompt: Role: DevOps Engineer | Task: Create config.yaml for CFST XGBoost pipeline with all necessary parameters following the design in design.md section "Data Models" | Restrictions: YAML format must be valid, include all required sections, use appropriate data types | Success: config.yaml created with data, model, cv, and paths sections, all parameters properly configured

- [x] 1.3. 创建依赖文件 requirements.txt
  - 列出所有Python依赖包及版本
  - 包括 xgboost, pandas, numpy, scikit-learn, optuna等
  - _Prompt: Role: Python Developer | Task: Create requirements.txt with all dependencies for CFST XGBoost pipeline project | Restrictions: Specify compatible versions, include both direct and indirect dependencies | Success: requirements.txt created, all packages listed with appropriate versions

- [x] 1.4. 设置日志配置
  - File: `src/utils/logger.py`
  - 配置日志格式、级别、输出位置
  - 支持控制台和文件输出
  - _Prompt: Role: Python Developer | Task: Create logger utility module for CFST XGBoost pipeline with proper configuration | Restrictions: Use Python logging module, support configurable log levels, format consistent | Success: Logger module created, logs properly formatted, both console and file output work

---

## Task 2: 实现数据加载和预处理模块

- [x] 2.1. 实现数据加载器 DataLoader
  - File: `src/data_loader.py`
  - 从CSV加载数据，识别特征和标签列
  - 验证数据完整性，检查必要列是否存在
  - _Prompt: Role: Data Engineer | Task: Implement DataLoader class for CFST XGBoost pipeline following design.md Component 1 specifications | Restrictions: Must handle file not found errors, validate column existence, return proper data structures | Success: DataLoader loads CSV correctly, validates columns, returns features and target, error handling works
  - _Requirements: Requirements 1 (数据加载与预处理)

- [x] 2.2. 实现预处理器 Preprocessor
  - File: `src/preprocessor.py`
  - 剔除指定列 b, h, r₀, t, L, λ
  - 处理缺失值（检查是否需要填充或删除）
  - _Prompt: Role: Data Engineer | Task: Implement Preprocessor class to drop specified columns and handle missing values per design.md Component 2 | Restrictions: Must drop exactly b, h, r0, t, L, lambda columns, preserve all others, handle missing data appropriately | Success: Preprocessor drops correct columns, handles missing values, returns clean feature set, get_remaining_features works correctly
  - _Requirements: Requirements 1 (数据加载与预处理)

- [ ] 2.3. 测试数据加载和预处理
  - File: `tests/test_data_loader.py`, `tests/test_preprocessor.py`
  - 创建单元测试，覆盖正常和异常场景
  - 测试缺失文件、缺失列、数据格式错误等
  - _Prompt: Role: QA Engineer | Task: Create comprehensive unit tests for DataLoader and Preprocessor components | Restrictions: Use pytest framework, cover edge cases, mock file operations appropriately | Success: All tests pass, coverage > 90%, edge cases handled, error scenarios tested
  - _Requirements: Design "Testing Strategy" - Unit Testing

---

## Task 3: 实现XGBoost模型训练模块

- [x] 3.1. 实现模型训练器 ModelTrainer
  - File: `src/model_trainer.py`
  - 实现XGBoost模型训练，支持自定义参数
  - 实现5折交叉验证
  - _Prompt: Role: ML Engineer | Task: Implement ModelTrainer class for XGBoost training with cross-validation per design.md Component 3 | Restrictions: Use xgboost.XGBRegressor, implement proper cross-validation, log training progress | Success: ModelTrainer trains XGBoost model correctly, cross-validation runs, returns trained model, logs training metrics
  - _Requirements: Requirements 2 (XGBoost模型训练)

- [x] 3.2. 集成Optuna超参数优化（可选）
  - File: `src/model_trainer.py` (延伸)
  - 实现Optuna超参数搜索
  - 支持n_trials配置，保存最优参数
  - _Prompt: Role: ML Engineer | Task: Integrate Optuna hyperparameter optimization into ModelTrainer when use_optuna=True | Restrictions: Only optimize when explicitly enabled, save best parameters, log optimization progress | Success: Optuna optimization works, finds better hyperparameters, optimization history logged
  - _Requirements: Requirements 2 (XGBoost模型训练)

- [x] 3.3. 实现模型工具函数
  - File: `src/utils/model_utils.py`
  - 保存模型、预处理器和元数据
  - 加载模型、预处理器和元数据
  - _Prompt: Role: ML Engineer | Task: Implement model persistence utilities for saving/loading models and preprocessors per design.md Component 7 | Restrictions: Use joblib for serialization, save metadata as JSON, include feature names and model parameters | Success: save_model saves all components correctly, load_model loads everything properly, metadata complete
  - _Requirements: Requirements 5 (模型持久化与管理)

- [ ] 3.4. 测试模型训练功能
  - File: `tests/test_model_trainer.py`
  - 测试模型训练、交叉验证、超参数优化
  - 验证保存的模型可以正确加载
  - _Prompt: Role: QA Engineer | Task: Create unit tests for ModelTrainer and model persistence | Restrictions: Use small dataset for fast testing, mock expensive operations, clean up test outputs | Success: All training tests pass, model can be saved and loaded, cross-validation tested, optuna integration tested
  - _Requirements: Design "Testing Strategy" - Unit Testing

---

## Task 4: 实现模型评估模块

- [ ] 4.1. 实现评估器 Evaluator
  - File: `src/evaluator.py`
  - 计算RMSE、MAE、R²、MAPE等指标
  - 生成评估报告字典
  - _Prompt: Role: ML Engineer | Task: Implement Evaluator class with comprehensive metrics calculation per design.md Component 4 | Restrictions: Implement all four metrics (RMSE, MAE, R², MAPE), return formatted results, use numpy for calculations | Success: Evaluator calculates all metrics correctly, handles edge cases, returns properly formatted report
  - _Requirements: Requirements 3 (模型评估与验证)

- [ ] 4.2. 实现预测结果可视化
  - File: `src/visualizer.py`
  - 生成预测值vs真实值散点图
  - 保存图表到output目录
  - _Prompt: Role: Data Scientist | Task: Implement plot_predictions_scatter function to visualize predictions vs actual values | Restrictions: Use matplotlib, proper labels and title, save as PNG, include R² in title | Success: Scatter plot generated correctly, shows good correlation, saved to output directory
  - _Requirements: Requirements 3 (模型评估与验证)

- [ ] 4.3. 生成评估报告
  - File: `src/evaluator.py` (延伸)
  - 保存评估指标到JSON文件
  - 包含模型参数、特征列表、交叉验证结果
  - _Prompt: Role: ML Engineer | Task: Implement save_evaluation_report function to persist evaluation results | Restrictions: JSON format, include all metadata, timestamp, human-readable | Success: Evaluation report saved correctly, contains all metrics and metadata, can be loaded for later analysis
  - _Requirements: Requirements 3 (模型评估与验证)

- [ ] 4.4. 测试评估功能
  - File: `tests/test_evaluator.py`, `tests/test_visualizer.py`
  - 测试指标计算准确性
  - 测试图表生成
  - _Prompt: Role: QA Engineer | Task: Create tests for Evaluator and Visualizer components | Restrictions: Test metric calculations with known values, verify chart files are created, clean up test outputs | Success: All metric calculations correct, visualizations generated successfully, tests cover edge cases like division by zero in MAPE
  - _Requirements: Design "Testing Strategy" - Unit Testing

---

## Task 5: 实现XGBoost特征重要性可视化

- [ ] 5.1. 实现特征重要性可视化
  - File: `src/visualizer.py` (延伸)
  - 生成XGBoost特征重要性图（bar plot）
  - 保存图表文件
  - _Prompt: Role: Data Scientist | Task: Implement plot_feature_importance function using XGBRegressor's feature_importances_ attribute | Restrictions: Use matplotlib/seaborn, horizontal bar chart, sorted by importance, clear labels, save as PNG | Success: Feature importance plot generated, shows top features clearly, saved to output directory
  - _Requirements: Requirements 6 (数据可视化)

- [ ] 5.2. 实现特征重要性文本输出
  - File: `src/visualizer.py` (延伸)
  - 打印特征重要性排名到控制台
  - 保存排名到文本文件
  - _Prompt: Role: Python Developer | Task: Implement print_feature_importance_ranking function to display and save feature rankings | Restrictions: Sorted by importance desc, include feature name and importance value, formatted output | Success: Feature rankings displayed clearly, saved to text file, easy to interpret
  - _Requirements: Requirements 6 (数据可视化)

---

## Task 6: 实现预测模块

- [ ] 6.1. 实现预测器 Predictor
  - File: `src/predictor.py`
  - 加载训练好的模型和预处理器
  - 实现单条记录预测和批量预测
  - _Prompt: Role: ML Engineer | Task: Implement Predictor class for making predictions with trained XGBoost model following design.md Component 5 | Restrictions: Load model and preprocessor from files, validate input format matches training data, support both single and batch prediction | Success: Predictor loads model correctly, validates inputs, returns accurate predictions, batch prediction efficient
  - _Requirements: Requirements 4 (模型预测功能)

- [ ] 6.2. 实现预测结果导出
  - File: `src/predictor.py` (延伸)
  - 将预测结果保存为CSV文件
  - 包含输入特征和预测值
  - _Prompt: Role: Python Developer | Task: Implement export_predictions function to save prediction results with features | Restrictions: CSV format, include all input features and prediction column, proper headers | Success: Predictions exported correctly, CSV format valid, can be opened in Excel or other tools
  - _Requirements: Requirements 4 (模型预测功能)

- [ ] 6.3. 测试预测功能
  - File: `tests/test_predictor.py`
  - 测试单条预测、批量预测
  - 测试预测结果导出
  - _Prompt: Role: QA Engineer | Task: Create tests for Predictor component covering various prediction scenarios | Restrictions: Use saved test model, verify prediction shapes and values, test error handling for invalid inputs | Success: All prediction tests pass, batch prediction works efficiently, error handling tested
  - _Requirements: Design "Testing Strategy" - Unit Testing

---

## Task 7: 创建主脚本和命令行接口

- [ ] 7.1. 创建训练脚本 train.py
  - File: `train.py`
  - 命令行接口：配置路径作为参数
  - 执行完整训练流程：加载数据→预处理→训练→评估→保存
  - _Prompt: Role: Python Developer | Task: Create train.py script that executes complete training pipeline | Restrictions: Accept config path as argument, use logging, handle errors gracefully, save all outputs | Success: train.py runs complete pipeline, creates all outputs, logs progress, handles errors
  - _Requirements: Requirements 1-5

- [ ] 7.2. 创建预测脚本 predict.py
  - File: `predict.py`
  - 命令行接口：模型路径、输入数据路径、输出路径
  - 执行预测流程：加载模型→加载数据→预测→导出结果
  - _Prompt: Role: Python Developer | Task: Create predict.py script for making predictions with saved model | Restrictions: Accept model path, input data path, output path as arguments, validate inputs, export results | Success: predict.py loads model correctly, makes predictions, exports results to CSV
  - _Requirements: Requirements 4-5

---

## Task 8: 集成测试和端到端测试

- [ ] 8.1. 创建集成测试
  - File: `tests/test_integration.py`
  - 测试数据加载→预处理→训练→评估完整流程
  - 验证各模块协同工作
  - _Prompt: Role: QA Engineer | Task: Create integration tests for complete ML pipeline | Restrictions: Test full workflow end-to-end, use real data, verify all outputs created, clean up after tests | Success: Integration tests pass, all components work together, outputs validated
  - _Requirements: Design "Testing Strategy" - Integration Testing

- [ ] 8.2. 创建端到端测试
  - File: `tests/test_end_to_end.py`
  - 使用真实CFST数据训练模型
  - 验证跨截面预测能力
  - 性能基准测试
  - _Prompt: Role: QA Engineer | Task: Create end-to-end tests with real CFST data | Restrictions: Use actual feature_parameters.csv, verify model performance meets expectations, test prediction speed | Success: E2E tests demonstrate model works on real data, performance acceptable, predictions reasonable
  - _Requirements: Design "Testing Strategy" - End-to-End Testing

---

## Task 9: 文档和清理

- [ ] 9.1. 创建README文档
  - File: `README.md`
  - 项目介绍、安装步骤、使用说明
  - 参数说明、示例命令
  - _Prompt: Role: Technical Writer | Task: Create comprehensive README for CFST XGBoost pipeline | Restrictions: Include setup, usage examples, parameter descriptions, example commands | Success: README complete, clear instructions, covers all use cases

- [ ] 9.2. 添加代码注释和文档字符串
  - 为所有函数添加docstring
  - 添加关键代码注释
  - 遵循Google或NumPy风格
  - _Prompt: Role: Python Developer | Task: Add comprehensive docstrings and comments to all code | Restrictions: Google/NumPy docstring style, explain parameters, returns, examples | Success: All functions documented, code is readable and maintainable

- [ ] 9.3. 创建示例配置文件
  - File: `config/config.example.yaml`
  - 包含所有参数说明
  - 作为配置模板
  - _Prompt: Role: DevOps Engineer | Task: Create example configuration file with documentation | Restrictions: All parameters explained, good default values, well-commented | Success: Example config is clear and complete, can be used as template

---

## Task 10: 运行完整管道并生成结果

- [ ] 10.1. 运行训练管道
  - 使用实际feature_parameters.csv数据
  - 训练XGBoost模型
  - 生成评估报告和可视化
  - _Prompt: Role: ML Engineer | Task: Run complete training pipeline on actual CFST data | Restrictions: Use real data, capture all outputs, document results | Success: Model trained successfully, evaluation metrics calculated, plots generated, results saved
  - _Requirements: All Requirements 1-6

- [ ] 10.2. 验证模型性能
  - 检查评估指标（RMSE, MAE, R², MAPE）
  - 分析特征重要性
  - 验证预测准确性
  - _Prompt: Role: ML Engineer | Task: Analyze and validate model performance metrics | Restrictions: Check if metrics meet expectations, interpret feature importance, assess practical significance | Success: Performance acceptable, feature importance makes engineering sense, model is reliable

- [ ] 10.3. 生成最终报告
  - 汇总所有结果
  - 创建模型性能总结
  - 提供模型使用说明
  - _Prompt: Role: Data Scientist | Task: Create comprehensive final report with all results and analysis | Restrictions: Include metrics, feature importance, predictions vs actual, model specifications | Success: Final report complete, professional format, all stakeholders can understand results
