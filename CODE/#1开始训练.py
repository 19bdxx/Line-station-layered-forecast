import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*deprecated.*")


# -------------------- 代码说明 --------------------
# 本代码旨在通过使用多种模型预测风电场的功率，并计算多个评估指标（RMSE, MAE, MAPE, R² 和偏差率）。
# 支持的模型类型（MODEL_TYPE）：
#   'lightgbm'     - LightGBM 梯度提升树
#   'xgboost'      - XGBoost 梯度提升树
#   'random_forest'- 随机森林（scikit-learn）
#   'ridge'        - 岭回归（scikit-learn）
#   'mlp'          - 多层感知机神经网络（scikit-learn MLPRegressor）
#   'lstm'         - 长短期记忆网络（PyTorch）
# 具体步骤如下：
# 1. 加载数据并进行预处理；
# 2. 针对不同的风电场、目标列和预测步长进行多次实验；
# 3. 根据 MODEL_TYPE 训练对应模型，计算预测值；
# 4. 计算评估指标，包括 RMSE, MAE, MAPE, R² 和偏差率；
# 5. 将每次实验的结果保存为 CSV 文件，并输出到控制台。

# -------------------- 配置 --------------------
M = 32
# 自动依次训练以下所有模型；如只需运行部分模型，注释掉不需要的条目即可。
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm']
DATA_PATH = r"G:\WindPowerForecast\集群_场站预测\data\all_stations_输电线未发电置零_after20240718_15min.csv"
predict_steps = [i for i in range(1, 96)]

stations = {
    "XYA": {
        "target_cols": ['XYA_ACTIVE_POWER_JIA','XYA_ACTIVE_POWER_YI','XYA_ACTIVE_POWER_STATION'],
        "limit_col": 'XYA_LIMIT_POWER',
        "P_capacity": 400
    },
    "XYB": {
        "target_cols": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU', 'XYB_ACTIVE_POWER_STATION'],
        "limit_col": 'XYB_LIMIT_POWER',
        "P_capacity": 900
    },
    "XS": {
        "target_cols": ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI', 'XS_ACTIVE_POWER_STATION'],
        "limit_col": 'XS_LIMIT_POWER',
        "P_capacity": 400
    }
}

# -------------------- 模型构建与训练 --------------------
def build_and_train_model(model_type, X_tr, y_tr, X_val, y_val, seq_len, n_features):
    """
    根据 model_type 构建并训练模型。
    返回 (model, evals_result)，其中 evals_result 仅 LightGBM 有值，其余为 None。
    """
    if model_type == 'lightgbm':
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1
        }
        evals_result = {}
        model = lgb.train(
            params, train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(50),
                record_evaluation(evals_result)
            ]
        )
        return model, evals_result

    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=100,
            eval_metric='rmse',
            verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model, None

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        return model, None

    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        model.fit(X_tr, y_tr)
        return model, None

    elif model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
        model.fit(X_tr, y_tr)
        return model, None

    elif model_type == 'lstm':
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        def to_3d(X):
            return X.reshape(X.shape[0], seq_len, n_features).astype(np.float32)

        X_tr_3d = to_3d(X_tr)
        X_val_3d = to_3d(X_val)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tr_ds = TensorDataset(torch.from_numpy(X_tr_3d), torch.from_numpy(y_tr.astype(np.float32)))
        val_ds = TensorDataset(torch.from_numpy(X_val_3d), torch.from_numpy(y_val.astype(np.float32)))
        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        net = LSTMRegressor(input_size=n_features).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience, no_improve = 20, 0
        best_state = None

        for epoch in range(200):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(net(xb), yb).item() * len(xb)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        net.load_state_dict(best_state)
        return net, None

    else:
        raise ValueError(f"不支持的模型类型：{model_type}。"
                         f"可选值：'lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm'")


def predict_with_model(model, model_type, X, seq_len, n_features):
    """使用训练好的模型进行预测，返回预测数组。"""
    if model_type == 'lightgbm':
        return model.predict(X, num_iteration=model.best_iteration)
    elif model_type == 'lstm':
        import torch
        device = next(model.parameters()).device
        X_3d = X.reshape(X.shape[0], seq_len, n_features).astype(np.float32)
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_3d).to(device)).cpu().numpy()
        return preds
    else:
        return model.predict(X)


# -------------------- 计算偏差率 --------------------
def calculate_bias_rate(y_true, y_pred, P_capacity):
    bias_rate = np.zeros_like(y_true)
    condition = y_true >= 0.2 * P_capacity
    bias_rate[condition] = abs(y_pred[condition] - y_true[condition]) / y_true[condition]
    bias_rate[~condition] = abs(y_pred[~condition] - y_true[~condition]) / (0.2 * P_capacity)
    return bias_rate

# -------------------- 绘图函数 --------------------
def plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir):
    train_rmse = evals_result['train']['rmse']
    valid_rmse = evals_result['valid']['rmse']
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(valid_rmse, label='Valid RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'{station_name} | {target_col} | t+{predict_step} RMSE Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图像
    plot_path = os.path.join(save_dir, f"{target_col}_t+{predict_step}_RMSE_curve.png")
    plt.savefig(plot_path)
    plt.close()

# -------------------- 主实验函数 --------------------
def run_experiment(df, station_name, target_col, predict_step, limit_col, include_limit, save_dir, model_type):
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")
    hist_features = [target_col]
    if include_limit:
        hist_features.append(limit_col)

    n_features = len(hist_features)
    seq_len = M + 1  # 历史窗口长度（含当前时刻）

    X_list, y_list, ts_list = [], [], []

    for i in range(M, len(df) - predict_step):
        x_hist = df[hist_features].iloc[i - M:i + 1].values.flatten()
        x_input = x_hist
        y_target = df.loc[i + predict_step, target_col]
        ts_target = df.loc[i + predict_step, 'timestamp']
        P_capacity = stations[station_name]['P_capacity']

        X_list.append(x_input)
        y_list.append(y_target)
        ts_list.append(ts_target)

    X = np.array(X_list)
    y = np.array(y_list)
    ts = np.array(ts_list)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts[split_idx:]

    split_idx_train = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val = y_train[:split_idx_train], y_train[split_idx_train:]

    model, evals_result = build_and_train_model(
        model_type, X_tr, y_tr, X_val, y_val, seq_len, n_features
    )

    y_pred = predict_with_model(model, model_type, X_test, seq_len, n_features)
    bias_rate = calculate_bias_rate(y_test, y_pred, P_capacity)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'bias_rate': bias_rate,
        'limit_value': [df.loc[i + predict_step, limit_col] for i in range(len(y_test))]
    })

    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    # 仅 LightGBM 支持 RMSE 曲线图
    if model_type == 'lightgbm' and evals_result:
        plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir)

    print(f"\n✅ {station_name} {'含限电' if include_limit else '无限电'} | {file_prefix} [{model_type}]")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    return {
        'station': station_name,
        'model': model_type,
        'limit_mode': 'with_limit' if include_limit else 'no_limit',
        'target': target_col,
        'step': predict_step,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

# -------------------- 执行所有实验 --------------------
df_all = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
all_results = []

for model_type in MODEL_TYPES:
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型：{model_type}")
    print(f"{'='*60}")
    model_results = []

    try:
        for station_name, config in stations.items():
            for include_limit in [True, False]:
                sub_dir = '对比实验' if include_limit else '对比实验_无限电'
                save_dir = os.path.join(sub_dir, model_type, station_name)
                os.makedirs(save_dir, exist_ok=True)

                for target_col in config['target_cols']:
                    for step in predict_steps:
                        result = run_experiment(
                            df_all, station_name, target_col, step,
                            config['limit_col'], include_limit, save_dir, model_type
                        )
                        model_results.append(result)
                        all_results.append(result)

        summary_df = pd.DataFrame(model_results)
        summary_df.to_csv(f'所有实验汇总_metrics_{model_type}.csv', index=False)
        print(f"\n📊 [{model_type}] 实验完成，汇总结果已保存为 所有实验汇总_metrics_{model_type}.csv")

    except Exception as e:
        import traceback
        print(f"\n❌ 模型 [{model_type}] 执行出错，已跳过。错误信息：{e}")
        traceback.print_exc()
        continue

all_summary_df = pd.DataFrame(all_results)
all_summary_df.to_csv('所有实验汇总_metrics_ALL.csv', index=False)
print(f"\n🎉 全部模型实验完成！综合汇总已保存为 所有实验汇总_metrics_ALL.csv")