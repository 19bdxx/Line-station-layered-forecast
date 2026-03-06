import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*deprecated.*")


# -------------------- 代码说明 --------------------
# 本脚本是 #1开始训练.py 的升级版，核心改进：
#
#   将同一场站内所有线路的历史数据（含目标线路本身）同时作为特征输入，
#   以显式利用线路间的相关性来提升预测精度。
#
# 与原脚本 (#1) 的对比：
#   - 原脚本：每条线路独立建模，仅用该线路自身的历史数据作特征。
#   - 本脚本：对同一场站内每条线路建模时，额外引入场站内所有其他线路
#             的同期历史数据作为补充特征（跨线路历史特征增强）。
#
# 以 XYA 场站为例（甲线 + 乙线）：
#   - 预测甲线：特征 = [甲线历史, 乙线历史] （+ 可选限电）
#   - 预测乙线：特征 = [乙线历史, 甲线历史] （+ 可选限电）
#   - 预测全站：特征 = [全站历史, 甲线历史, 乙线历史] （+ 可选限电）
#
# 支持的模型类型（MODEL_TYPE）：
#   'lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm'
# 与 #1 脚本完全相同，可直接切换。
#
# 输出目录：
#   跨线路增强实验/{model_type}/{station}/  ← 含限电
#   跨线路增强实验_无限电/{model_type}/{station}/  ← 无限电
#   所有实验汇总_crossline_metrics_{model_type}.csv

# -------------------- 配置（与 #1 保持一致，修改 DATA_PATH 即可） --------------------
M = 32
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm']
DATA_PATH = r"G:\WindPowerForecast\集群_场站预测\data\all_stations_输电线未发电置零_after20240718_15min.csv"
predict_steps = [i for i in range(1, 96)]

# 场站配置：
#   all_line_cols  —— 该场站下所有有功功率列（子线路 + 全站），用于构造跨线路特征
#   target_cols    —— 训练目标列（与 #1 相同）
#   limit_col      —— 限电字段
#   P_capacity     —— 装机容量（MW），用于计算偏差率
stations = {
    "XYA": {
        "all_line_cols": [
            'XYA_ACTIVE_POWER_JIA',
            'XYA_ACTIVE_POWER_YI',
            'XYA_ACTIVE_POWER_STATION',
        ],
        "target_cols": [
            'XYA_ACTIVE_POWER_JIA',
            'XYA_ACTIVE_POWER_YI',
            'XYA_ACTIVE_POWER_STATION',
        ],
        "limit_col": 'XYA_LIMIT_POWER',
        "P_capacity": 400,
    },
    "XYB": {
        "all_line_cols": [
            'XYB_ACTIVE_POWER_BING',
            'XYB_ACTIVE_POWER_DING',
            'XYB_ACTIVE_POWER_WU',
            'XYB_ACTIVE_POWER_STATION',
        ],
        "target_cols": [
            'XYB_ACTIVE_POWER_BING',
            'XYB_ACTIVE_POWER_DING',
            'XYB_ACTIVE_POWER_WU',
            'XYB_ACTIVE_POWER_STATION',
        ],
        "limit_col": 'XYB_LIMIT_POWER',
        "P_capacity": 900,
    },
    "XS": {
        "all_line_cols": [
            'XS_ACTIVE_POWER_JIA',
            'XS_ACTIVE_POWER_YI',
            'XS_ACTIVE_POWER_STATION',
        ],
        "target_cols": [
            'XS_ACTIVE_POWER_JIA',
            'XS_ACTIVE_POWER_YI',
            'XS_ACTIVE_POWER_STATION',
        ],
        "limit_col": 'XS_LIMIT_POWER',
        "P_capacity": 400,
    },
}


# -------------------- 模型构建与训练（与 #1 完全相同） --------------------
def build_and_train_model(model_type, X_tr, y_tr, X_val, y_val, seq_len, n_features):
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
            'verbose': -1,
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
                record_evaluation(evals_result),
            ],
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
            verbosity=0,
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
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        return model, None

    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
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
                random_state=42,
            )),
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
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        def to_3d(X):
            return X.reshape(X.shape[0], seq_len, n_features).astype(np.float32)

        X_tr_3d  = to_3d(X_tr)
        X_val_3d = to_3d(X_val)

        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tr_ds     = TensorDataset(torch.from_numpy(X_tr_3d),  torch.from_numpy(y_tr.astype(np.float32)))
        val_ds    = TensorDataset(torch.from_numpy(X_val_3d), torch.from_numpy(y_val.astype(np.float32)))
        tr_loader  = DataLoader(tr_ds,  batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        net        = LSTMRegressor(input_size=n_features).to(device)
        optimizer  = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion  = nn.MSELoss()

        best_val_loss = float('inf')
        patience, no_improve = 20, 0
        best_state = None

        for epoch in range(200):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(net(xb), yb).backward()
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
                best_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        net.load_state_dict(best_state)
        return net, None

    else:
        raise ValueError(
            f"不支持的模型类型：{model_type}。"
            f"可选值：'lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm'"
        )


def predict_with_model(model, model_type, X, seq_len, n_features):
    if model_type == 'lightgbm':
        return model.predict(X, num_iteration=model.best_iteration)
    elif model_type == 'lstm':
        import torch
        device = next(model.parameters()).device
        X_3d   = X.reshape(X.shape[0], seq_len, n_features).astype(np.float32)
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_3d).to(device)).cpu().numpy()
        return preds
    else:
        return model.predict(X)


def calculate_bias_rate(y_true, y_pred, P_capacity):
    bias_rate = np.zeros_like(y_true)
    condition  = y_true >= 0.2 * P_capacity
    bias_rate[condition]  = abs(y_pred[condition]  - y_true[condition])  / y_true[condition]
    bias_rate[~condition] = abs(y_pred[~condition] - y_true[~condition]) / (0.2 * P_capacity)
    return bias_rate


# -------------------- 核心改动：跨线路特征增强实验函数 --------------------
def run_experiment_crossline(
    df, station_name, target_col, predict_step,
    cross_feature_cols,     # ← 新增：同场站所有有功功率列（含目标列）
    limit_col, include_limit, save_dir, model_type
):
    """
    与 run_experiment（#1脚本）的唯一区别：
        hist_features 由"仅目标列"改为"同场站所有有功功率列"，
        从而让模型能学习到线路间的相关模式。

    参数
    ----
    cross_feature_cols : list[str]
        同场站所有有功功率列名，例如：
        ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI', 'XYA_ACTIVE_POWER_STATION']
        预测目标 target_col 也必须包含在其中，否则抛出 ValueError。
    """
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")
    if target_col not in cross_feature_cols:
        raise ValueError(f"target_col [{target_col}] 必须包含在 cross_feature_cols 中")

    # 历史特征 = 所有线路历史 + 可选限电
    # 目标列排在第一位，其余线路列追加在后，保持特征顺序的一致性
    ordered_features = [target_col] + [c for c in cross_feature_cols if c != target_col]
    hist_features    = ordered_features + ([limit_col] if include_limit else [])

    n_features = len(hist_features)
    seq_len    = M + 1   # 历史窗口长度（含当前时刻）

    X_list, y_list, ts_list = [], [], []

    for i in range(M, len(df) - predict_step):
        # 展平 (seq_len × n_features) 的历史窗口
        x_hist   = df[hist_features].iloc[i - M:i + 1].values.flatten()
        y_target = df.loc[i + predict_step, target_col]
        ts_val   = df.loc[i + predict_step, 'timestamp']
        X_list.append(x_hist)
        y_list.append(y_target)
        ts_list.append(ts_val)

    X  = np.array(X_list)
    y  = np.array(y_list)
    ts = np.array(ts_list)

    split_idx     = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test         = ts[split_idx:]

    split_idx_train = int(len(X_train) * 0.8)
    X_tr, X_val    = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val_   = y_train[:split_idx_train], y_train[split_idx_train:]

    model, _ = build_and_train_model(
        model_type, X_tr, y_tr, X_val, y_val_, seq_len, n_features
    )

    y_pred    = predict_with_model(model, model_type, X_test, seq_len, n_features)
    P_cap     = stations[station_name]['P_capacity']
    bias_rate = calculate_bias_rate(y_test, y_pred, P_cap)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2   = r2_score(y_test, y_pred)

    results_df = pd.DataFrame({
        'timestamp':   ts_test,
        'y_true':      y_test,
        'y_pred':      y_pred,
        'bias_rate':   bias_rate,
        'limit_value': [df.loc[split_idx + i + predict_step, limit_col]
                        for i in range(len(y_test))],
    })

    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    lim_tag = '含限电' if include_limit else '无限电'
    print(f"\n✅ {station_name} {lim_tag} | {file_prefix} [{model_type}] [跨线路增强]")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    return {
        'station':    station_name,
        'model':      model_type,
        'limit_mode': 'with_limit' if include_limit else 'no_limit',
        'target':     target_col,
        'step':       predict_step,
        'rmse':       rmse,
        'mae':        mae,
        'mape':       mape,
        'r2':         r2,
    }


# -------------------- 执行所有实验 --------------------
df_all     = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
all_results = []

for model_type in MODEL_TYPES:
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型（跨线路增强）：{model_type}")
    print(f"{'='*60}")
    model_results = []

    try:
        for station_name, config in stations.items():
            for include_limit in [True, False]:
                sub_dir  = '跨线路增强实验' if include_limit else '跨线路增强实验_无限电'
                save_dir = os.path.join(sub_dir, model_type, station_name)
                os.makedirs(save_dir, exist_ok=True)

                for target_col in config['target_cols']:
                    for step in predict_steps:
                        result = run_experiment_crossline(
                            df_all, station_name, target_col, step,
                            config['all_line_cols'],
                            config['limit_col'], include_limit,
                            save_dir, model_type,
                        )
                        model_results.append(result)
                        all_results.append(result)

        summary_df = pd.DataFrame(model_results)
        out_csv    = f'所有实验汇总_crossline_metrics_{model_type}.csv'
        summary_df.to_csv(out_csv, index=False)
        print(f"\n📊 [{model_type}] 跨线路实验完成，汇总已保存为 {out_csv}")

    except Exception as e:
        import traceback
        print(f"\n❌ 模型 [{model_type}] 执行出错，已跳过。错误信息：{e}")
        traceback.print_exc()
        continue

all_summary_df = pd.DataFrame(all_results)
all_summary_df.to_csv('所有实验汇总_crossline_metrics_ALL.csv', index=False)
print(f"\n🎉 跨线路增强实验全部完成！综合汇总已保存为 所有实验汇总_crossline_metrics_ALL.csv")
