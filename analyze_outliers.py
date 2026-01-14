"""
独立的离群点分析脚本
不影响现有代码，可随时删除

功能：
1. 加载已训练的模型
2. 重现数据处理流程（使用相同的random_state）
3. 分析预测误差，找出离群点
4. 追踪到原始CSV文件的具体行号
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path


def analyze_outliers(
    model_path='output/xgboost_model.pkl',
    config_path='config/config.yaml',
    raw_data_path='data/raw/feature_parameters_unique.csv',
    output_path='output/outlier_analysis.csv'
):
    """
    加载模型和数据，分析预测误差中的离群点
    能够追踪到原始CSV文件的行号

    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        raw_data_path: 原始数据CSV路径
        output_path: 输出结果CSV路径

    Returns:
        results_df: 包含详细误差分析的DataFrame
    """

    # ============ Step 1: 加载配置 ============
    print("Loading configuration...")
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    target_column = data_config['target_column']
    columns_to_drop = data_config['columns_to_drop']
    test_size = data_config.get('test_size', 0.2)
    random_state = data_config.get('random_state', 42)
    target_transform_enabled = data_config['target_transform'].get('enabled', False)

    print(f"  Target column: {target_column}")
    print(f"  Test size: {test_size}")
    print(f"  Random state: {random_state}")
    print(f"  Target transform enabled: {target_transform_enabled}")

    # ============ Step 2: 加载原始数据并添加行号追踪 ============
    print(f"\nLoading raw data from {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path)  # CSV有表头

    # 添加CSV行号（1-based，对应文件行号，包含表头行则+1）
    # 实际文件中：第1行是表头，第2行开始是数据
    # 所以数据行号 = 原始索引 + 2
    df_raw['csv_row_num'] = df_raw.index + 2  # +1 for 1-based, +1 for header row

    print(f"  Loaded {len(df_raw)} rows")
    print(f"  Columns: {list(df_raw.columns)[:5]}...")

    # ============ Step 3: 分离特征和目标（与 data_loader.py 一致） ============
    features_full = df_raw.drop(columns=[target_column])
    target = df_raw[target_column]

    # 分离出需要追踪的行号
    csv_row_nums = df_raw['csv_row_num']

    print(f"\nFeatures before preprocessing: {len(features_full.columns)}")

    # ============ Step 4: 删除指定列（与 preprocessor.py 一致） ============
    features = features_full.drop(columns=columns_to_drop + ['csv_row_num'])

    print(f"Features after dropping: {len(features.columns)}")

    # ============ Step 5: 重新分割数据（使用相同的 random_state） ============
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")

    # 同时分割特征、目标值和行号
    X_train, X_test, y_train, y_test, row_num_train, row_num_test = train_test_split(
        features, target, csv_row_nums,
        test_size=test_size,
        random_state=random_state
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # ============ Step 6: 加载模型并预测 ============
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)

    # 加载预处理器（分开保存）
    preprocessor_path = Path(model_path).parent / 'preprocessor.pkl'
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)

    # 预处理测试集（与训练时一致）
    print("Preprocessing test set...")
    X_test_processed = preprocessor.transform(X_test)

    # 预测（对数空间）
    print("Making predictions...")
    y_pred_trans = model.predict(X_test_processed)

    # 逆变换到原始空间
    if target_transform_enabled:
        print("Applying inverse transform (exp)...")
        y_pred_orig = np.exp(y_pred_trans)
        y_test_orig = y_test.values  # y_test 已经是原始值
    else:
        y_pred_orig = y_pred_trans
        y_test_orig = y_test.values

    # ============ Step 7: 创建详细结果DataFrame ============
    print("\nCreating results DataFrame...")

    results_df = pd.DataFrame({
        'csv_row_num': row_num_test.values,        # 原始CSV文件行号
        'actual': y_test_orig,
        'predicted': y_pred_orig,
        'error': y_test_orig - y_pred_orig,
        'abs_error': np.abs(y_test_orig - y_pred_orig),
    })

    # 计算相对误差（%），避免除零
    with np.errstate(divide='ignore', invalid='ignore'):
        results_df['rel_error_pct'] = (results_df['error'] / results_df['actual'] * 100)
        results_df['rel_error_pct'] = results_df['rel_error_pct'].replace([np.inf, -np.inf], np.nan)

    # 按绝对误差排序
    results_df = results_df.sort_values('abs_error', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # ============ Step 8: 保存并输出结果 ============
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 输出统计信息
    print("\n" + "="*60)
    print("ERROR STATISTICS")
    print("="*60)
    print(f"Mean Absolute Error: {results_df['abs_error'].mean():.2f}")
    print(f"Median Absolute Error: {results_df['abs_error'].median():.2f}")
    print(f"Max Absolute Error: {results_df['abs_error'].max():.2f}")
    print(f"Std Dev of Absolute Error: {results_df['abs_error'].std():.2f}")

    # 输出前10个离群点
    print("\n" + "="*60)
    print("TOP 10 OUTLIERS (HIGHEST ABSOLUTE ERRORS)")
    print("="*60)
    print(results_df[['csv_row_num', 'actual', 'predicted', 'abs_error', 'rel_error_pct']].head(10).to_string())

    # 输出最大误差详情
    worst = results_df.iloc[0]
    print("\n" + "="*60)
    print("MAX ERROR DETAILS")
    print("="*60)
    print(f"CSV Row Number: {int(worst['csv_row_num'])}")
    print(f"Actual: {worst['actual']:.2f}")
    print(f"Predicted: {worst['predicted']:.2f}")
    print(f"Error: {worst['error']:.2f}")
    print(f"Absolute Error: {worst['abs_error']:.2f}")
    if pd.notna(worst['rel_error_pct']):
        print(f"Relative Error: {worst['rel_error_pct']:.2f}%")

    print(f"\nTo view this sample in original CSV:")
    row_num = int(worst['csv_row_num'])
    print(f"  sed -n '{row_num}p' {raw_data_path}")
    print(f"Or with headers:")
    print(f"  head -1 {raw_data_path}")
    print(f"  sed -n '{row_num}p' {raw_data_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return results_df


if __name__ == '__main__':
    analyze_outliers()
