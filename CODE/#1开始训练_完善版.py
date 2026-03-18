import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings



# -------------------- 代码说明 --------------------
# 本代码旨在通过使用多种模型预测风电场的功率，并计算多个评估指标（RMSE, MAE, MAPE, R² 和偏差率）。
# 支持的模型类型（MODEL_TYPE）：
#   'lightgbm'      - LightGBM 梯度提升树（支持 GPU，自动回退 CPU）
#   'xgboost'       - XGBoost 梯度提升树（支持 GPU，自动回退 CPU）
#   'random_forest' - 随机森林（scikit-learn，CPU）
#   'ridge'         - 岭回归（scikit-learn，CPU）
#   'mlp'           - 多层感知机神经网络（scikit-learn MLPRegressor，CPU）
#   'lstm'          - 长短期记忆网络（PyTorch，自动 CUDA）


# -------------------- 配置 --------------------
M = 32

# 自动依次训练以下所有模型；如只需运行部分模型，注释掉不需要的条目即可。
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp']
MODEL_TYPES = ['mlp']


DATA_PATH = r"G:\WindPowerForecast\集群_场站预测\data\all_stations_输电线未发电置零_after20240718_15min_mean.csv"
predict_steps = [i for i in range(1, 97)]

# 是否优先尝试 GPU
USE_GPU = True

stations = {
    "XYA": {
        "target_cols": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI', 'XYA_ACTIVE_POWER_STATION'],
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


# -------------------- GPU 检查 --------------------
def detect_torch_device():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print("=" * 60)
            print("🖥️ PyTorch GPU 状态")
            print("=" * 60)
            print(f"CUDA available: True")
            print(f"GPU count      : {torch.cuda.device_count()}")
            print(f"Current GPU    : {gpu_name}")
            try:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                print("cuDNN benchmark: True")
                print("float32 matmul : high")
            except Exception as perf_e:
                print(f"性能优化设置失败，但不影响训练。原因: {perf_e}")
            return device
        else:
            print("=" * 60)
            print("🖥️ PyTorch GPU 状态")
            print("=" * 60)
            print("CUDA available: False，LSTM 将使用 CPU")
            return torch.device("cpu")
    except Exception as e:
        print("=" * 60)
        print("🖥️ PyTorch GPU 状态")
        print("=" * 60)
        print(f"无法检测 torch/CUDA，LSTM 将使用 CPU。原因: {e}")
        return None


TORCH_DEVICE = detect_torch_device()


# -------------------- 工具函数 --------------------
def ensure_numeric_1d(arr, name):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} 为空")
    return arr


def ensure_numeric_2d(arr, name):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，当前 shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} 为空")
    return arr


def report_nonfinite(arr, name):
    arr = np.asarray(arr)
    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    return f"{name}: shape={arr.shape}, NaN={nan_count}, inf={inf_count}"



def compute_rmse(y_true, y_pred):
    """兼容不同 sklearn 版本的 RMSE 计算。"""
    y_true = ensure_numeric_1d(y_true, "y_true")
    y_pred = ensure_numeric_1d(y_pred, "y_pred")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def validate_train_arrays(X_tr, y_tr, X_val, y_val, model_type):
    X_tr = ensure_numeric_2d(X_tr, "X_tr")
    X_val = ensure_numeric_2d(X_val, "X_val")
    y_tr = ensure_numeric_1d(y_tr, "y_tr")
    y_val = ensure_numeric_1d(y_val, "y_val")

    # 所有模型都要求 y 不能有 NaN/inf
    if not np.isfinite(y_tr).all():
        raise ValueError("训练集标签异常: " + report_nonfinite(y_tr, "y_tr"))
    if not np.isfinite(y_val).all():
        raise ValueError("验证集标签异常: " + report_nonfinite(y_val, "y_val"))

    # 对不原生支持缺失值的模型，X 也必须全有限
    strict_models = {'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm'}
    if model_type in strict_models:
        if not np.isfinite(X_tr).all():
            raise ValueError("训练集特征异常: " + report_nonfinite(X_tr, "X_tr"))
        if not np.isfinite(X_val).all():
            raise ValueError("验证集特征异常: " + report_nonfinite(X_val, "X_val"))

    return X_tr, y_tr, X_val, y_val


def try_train_lightgbm_gpu(X_tr, y_tr, X_val, y_val):
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val)

    params_gpu = {
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
        'device': 'gpu'
    }

    evals_result = {}
    model = lgb.train(
        params_gpu,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(50),
            record_evaluation(evals_result)
        ]
    )
    return model, evals_result, "gpu"


def train_lightgbm_cpu(X_tr, y_tr, X_val, y_val):
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val)

    params_cpu = {
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
        'device': 'cpu'
    }

    evals_result = {}
    model = lgb.train(
        params_cpu,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(50),
            record_evaluation(evals_result)
        ]
    )
    return model, evals_result, "cpu"


def try_train_xgboost_gpu(X_tr, y_tr, X_val, y_val):
    import xgboost as xgb

    # 新版 XGBoost 推荐写法
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=100,
        eval_metric='rmse',
        verbosity=0,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, "gpu"


def train_xgboost_cpu(X_tr, y_tr, X_val, y_val):
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
        random_state=42,
        tree_method='hist',
        device='cpu'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, "cpu"


# -------------------- 模型构建与训练 --------------------
def build_and_train_model(model_type, X_tr, y_tr, X_val, y_val, seq_len, n_features):
    """
    根据 model_type 构建并训练模型。
    返回 (model, evals_result, device_info)
    """
    X_tr, y_tr, X_val, y_val = validate_train_arrays(X_tr, y_tr, X_val, y_val, model_type)

    if model_type == 'lightgbm':
        if USE_GPU:
            try:
                model, evals_result, used_device = try_train_lightgbm_gpu(X_tr, y_tr, X_val, y_val)
                print("✅ LightGBM 使用 GPU 训练")
                return model, evals_result, used_device
            except Exception as e:
                print(f"⚠️ LightGBM GPU 不可用，自动回退 CPU。原因: {e}")

        model, evals_result, used_device = train_lightgbm_cpu(X_tr, y_tr, X_val, y_val)
        print("✅ LightGBM 使用 CPU 训练")
        return model, evals_result, used_device

    elif model_type == 'xgboost':
        if USE_GPU:
            try:
                model, used_device = try_train_xgboost_gpu(X_tr, y_tr, X_val, y_val)
                print("✅ XGBoost 使用 GPU 训练")
                return model, None, used_device
            except Exception as e:
                print(f"⚠️ XGBoost GPU 不可用，自动回退 CPU。原因: {e}")

        model, used_device = train_xgboost_cpu(X_tr, y_tr, X_val, y_val)
        print("✅ XGBoost 使用 CPU 训练")
        return model, None, used_device

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
        print("✅ RandomForest 使用 CPU 训练")
        return model, None, "cpu"

    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        model.fit(X_tr, y_tr)
        print("✅ Ridge 使用 CPU 训练")
        return model, None, "cpu"

    elif model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
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
        print("✅ MLPRegressor 使用 CPU 训练（sklearn 不支持 GPU）")
        return model, None, "cpu"

    elif model_type == 'lstm':
        import time
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        def to_3d(X):
            return X.reshape(X.shape[0], seq_len, n_features).astype(np.float32)

        X_tr_3d = to_3d(X_tr)
        X_val_3d = to_3d(X_val)

        if not np.isfinite(X_tr_3d).all():
            raise ValueError("LSTM 训练集特征异常: " + report_nonfinite(X_tr_3d, "X_tr_3d"))
        if not np.isfinite(X_val_3d).all():
            raise ValueError("LSTM 验证集特征异常: " + report_nonfinite(X_val_3d, "X_val_3d"))

        device = TORCH_DEVICE if TORCH_DEVICE is not None else torch.device("cpu")
        use_cuda = (device.type == 'cuda')

        tr_ds = TensorDataset(
            torch.from_numpy(X_tr_3d),
            torch.from_numpy(y_tr.astype(np.float32))
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val_3d),
            torch.from_numpy(y_val.astype(np.float32))
        )

        if len(tr_ds) == 0 or len(val_ds) == 0:
            raise ValueError(f"LSTM 数据集为空: len(tr_ds)={len(tr_ds)}, len(val_ds)={len(val_ds)}")

        cpu_count = os.cpu_count() or 4
        num_workers = min(8, max(2, cpu_count // 2)) if use_cuda else 0
        batch_size = 1024 if use_cuda else 256
        pin_memory = use_cuda

        tr_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            drop_last=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            drop_last=False
        )

        net = LSTMRegressor(input_size=n_features).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        use_amp = use_cuda
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_val_loss = float('inf')
        patience, no_improve = 15, 0
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        max_epochs = 120

        print(f"✅ LSTM 使用 {'GPU' if use_cuda else 'CPU'} 训练")
        print(
            f"LSTM 配置: batch_size={batch_size}, num_workers={num_workers}, "
            f"amp={'on' if use_amp else 'off'}, hidden_size=128, max_epochs={max_epochs}"
        )

        for epoch in range(max_epochs):
            epoch_start = time.time()
            net.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = net(xb)
                    loss = criterion(pred, yb)

                if not torch.isfinite(loss):
                    raise ValueError(f"LSTM 训练 loss 非法: epoch={epoch}, loss={loss.item()}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item() * len(xb)
                train_count += len(xb)

            train_loss = train_loss_sum / max(train_count, 1)

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=use_cuda)
                    yb = yb.to(device, non_blocking=use_cuda)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        batch_loss = criterion(net(xb), yb).item()

                    if not np.isfinite(batch_loss):
                        raise ValueError(f"LSTM 验证 loss 非法: epoch={epoch}, batch_loss={batch_loss}")
                    val_loss += batch_loss * len(xb)

            val_loss /= len(val_ds)

            if not np.isfinite(val_loss):
                raise ValueError(f"LSTM val_loss 非法: epoch={epoch}, val_loss={val_loss}")

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"LSTM Epoch {epoch + 1:03d} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                    f"lr={current_lr:.6g} | {epoch_time:.2f}s"
                )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"LSTM Early stopping at epoch {epoch + 1}")
                    break

        net.load_state_dict(best_state)
        net.eval()
        return net, None, device.type

    else:
        raise ValueError(
            f"不支持的模型类型：{model_type}。"
            f"可选值：'lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm'"
        )


def predict_with_model(model, model_type, X, seq_len, n_features):
    """使用训练好的模型进行预测，返回预测数组。"""
    X = np.asarray(X, dtype=np.float32)

    if X.shape[0] == 0:
        return np.array([], dtype=np.float32)

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
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    bias_rate = np.zeros_like(y_true, dtype=np.float32)
    condition = y_true >= 0.2 * P_capacity
    bias_rate[condition] = np.abs(y_pred[condition] - y_true[condition]) / np.maximum(y_true[condition], 1e-8)
    bias_rate[~condition] = np.abs(y_pred[~condition] - y_true[~condition]) / (0.2 * P_capacity)
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

    plot_path = os.path.join(save_dir, f"{target_col}_t+{predict_step}_RMSE_curve.png")
    plt.savefig(plot_path)
    plt.close()


# -------------------- 主实验函数 --------------------
def run_experiment(df, station_name, target_col, predict_step, limit_col, include_limit, save_dir, model_type):
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")
    if limit_col not in df.columns:
        raise ValueError(f"未找到限电字段：{limit_col}")
    if 'timestamp' not in df.columns:
        raise ValueError("未找到 timestamp 字段")

    hist_features = [target_col]
    if include_limit:
        hist_features.append(limit_col)

    n_features = len(hist_features)
    seq_len = M + 1
    P_capacity = stations[station_name]['P_capacity']

    X_list, y_list, ts_list, limit_list = [], [], [], []

    for i in range(M, len(df) - predict_step):
        x_hist = df[hist_features].iloc[i - M:i + 1].values.flatten().astype(np.float32)
        y_target = df.loc[i + predict_step, target_col]
        ts_target = df.loc[i + predict_step, 'timestamp']
        limit_target = df.loc[i + predict_step, limit_col]

        if not np.isfinite(x_hist).all():
            continue
        if not np.isfinite(y_target):
            continue
        if not np.isfinite(limit_target):
            limit_target = np.nan

        X_list.append(x_hist)
        y_list.append(float(y_target))
        ts_list.append(ts_target)
        limit_list.append(float(limit_target) if np.isfinite(limit_target) else np.nan)

    if len(X_list) == 0:
        raise ValueError(f"{station_name}-{target_col}-t+{predict_step} 无有效样本，可能原始数据缺失过多")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ts = np.array(ts_list)
    limit_arr = np.array(limit_list, dtype=np.float32)

    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    ts = ts[valid_mask]
    limit_arr = limit_arr[valid_mask]

    if len(X) == 0:
        raise ValueError(f"{station_name}-{target_col}-t+{predict_step} 清洗后无有效样本")

    split_idx = int(len(X) * 0.8)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(f"{station_name}-{target_col}-t+{predict_step} 数据量不足，无法划分训练/测试集")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts[split_idx:]
    limit_test = limit_arr[split_idx:]

    split_idx_train = int(len(X_train) * 0.8)
    if split_idx_train <= 0 or split_idx_train >= len(X_train):
        raise ValueError(f"{station_name}-{target_col}-t+{predict_step} 数据量不足，无法划分训练/验证集")

    X_tr, X_val = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val = y_train[:split_idx_train], y_train[split_idx_train:]

    model, evals_result, used_device = build_and_train_model(
        model_type, X_tr, y_tr, X_val, y_val, seq_len, n_features
    )

    y_pred = predict_with_model(model, model_type, X_test, seq_len, n_features)

    if len(y_pred) != len(y_test):
        raise ValueError(
            f"预测长度与真实值长度不一致: len(y_pred)={len(y_pred)}, len(y_test)={len(y_test)}"
        )

    if not np.isfinite(y_pred).all():
        raise ValueError("预测结果含 NaN/inf: " + report_nonfinite(y_pred, "y_pred"))

    bias_rate = calculate_bias_rate(y_test, y_pred, P_capacity)

    rmse = compute_rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'bias_rate': bias_rate,
        'limit_value': limit_test
    })

    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    if model_type == 'lightgbm' and evals_result:
        plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir)

    print(f"\n✅ {station_name} {'含限电' if include_limit else '无限电'} | {file_prefix} [{model_type}]")
    print(f"训练设备: {used_device}")
    print(f"样本数 : total={len(X)}, train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"MAE   : {mae:.2f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"R²    : {r2:.4f}")

    return {
        'station': station_name,
        'model': model_type,
        'device': used_device,
        'limit_mode': 'with_limit' if include_limit else 'no_limit',
        'target': target_col,
        'step': predict_step,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


# -------------------- 数据读取与基础检查 --------------------
# 如果你的 CSV 之前报过坏行，可以临时加 on_bad_lines='skip'
df_all = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

print("=" * 60)
print("📋 原始数据基础检查")
print("=" * 60)
print(f"数据形状: {df_all.shape}")

num_df = df_all.select_dtypes(include=[np.number])
if num_df.shape[1] > 0:
    na_stat = df_all.isna().sum()
    na_stat = na_stat[na_stat > 0].sort_values(ascending=False)
    if len(na_stat) > 0:
        print("\n存在缺失值的字段（前20个）：")
        print(na_stat.head(20))
    else:
        print("\n未发现 NaN 缺失值")

    inf_count = int(np.isinf(num_df.to_numpy(dtype=np.float64)).sum())
    print(f"\n数值列 inf 总数: {inf_count}")
else:
    print("警告：未检测到数值列")

all_results = []

for model_type in MODEL_TYPES:
    print(f"\n{'=' * 60}")
    print(f"🚀 开始训练模型：{model_type}")
    print(f"{'=' * 60}")
    model_results = []

    try:
        for station_name, config in stations.items():
            for include_limit in [True, False]:
                sub_dir = '对比实验' if include_limit else '对比实验_无限电'
                save_dir = os.path.join(sub_dir, model_type, station_name)
                os.makedirs(save_dir, exist_ok=True)

                for target_col in config['target_cols']:
                    for step in predict_steps:
                        try:
                            result = run_experiment(
                                df_all,
                                station_name,
                                target_col,
                                step,
                                config['limit_col'],
                                include_limit,
                                save_dir,
                                model_type
                            )
                            model_results.append(result)
                            all_results.append(result)
                        except Exception as e_single:
                            print(
                                f"\n⚠️ 跳过单个实验: station={station_name}, "
                                f"target={target_col}, step={step}, "
                                f"include_limit={include_limit}, model={model_type}"
                            )
                            print(f"原因: {e_single}")
                            continue

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