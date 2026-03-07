import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- 配置 --------------------
# 自动运行的模型列表（与脚本 #1、#2 保持一致）
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm']

stations = {
    "XYA": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI'],
    "XYB": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU'],
    "XS":  ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI']
}

# -------------------- 比较函数 --------------------
def evaluate(y_true, y_pred):
    return {
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

# -------------------- 主循环 --------------------
all_results = []

for model_type in MODEL_TYPES:
    print(f"\n{'='*60}")
    print(f"📊 正在分析模型：{model_type}")
    print(f"{'='*60}")
    results = []

    try:
        base_dirs = {
            "with_limit": os.path.join("合并结果", model_type, "with_limit"),
            "no_limit":   os.path.join("合并结果", model_type, "no_limit")
        }

        for limit_mode, base_dir in base_dirs.items():
            if not os.path.exists(base_dir):
                print(f"⚠️ 跳过不存在的目录: {base_dir}")
                continue

            for station, sub_targets in stations.items():
                station_dir = os.path.join(base_dir, station)
                if not os.path.exists(station_dir):
                    print(f"⚠️ 跳过不存在的目录: {station_dir}")
                    continue

                for file in os.listdir(station_dir):
                    if not file.endswith("_merged.csv"):
                        continue

                    try:
                        step = int(file.split("_t+")[1].split("_")[0])
                    except:
                        print(f"⚠️ 文件名异常: {file}")
                        continue

                    file_path = os.path.join(station_dir, file)
                    df = pd.read_csv(file_path)

                    station_true_col = "STATION_true"
                    station_pred_col = "STATION_pred"

                    # 判断是否包含必要字段
                    if station_pred_col not in df.columns or station_true_col not in df.columns:
                        print(f"⚠️ 缺少全站字段，跳过: {file}")
                        continue

                    # 获取子线路预测列（不含 STATION 的）
                    sub_pred_cols = [col for col in df.columns if col.endswith("_pred") and "STATION" not in col]
                    if not sub_pred_cols:
                        print(f"⚠️ 没有子线路预测列，跳过: {file}")
                        continue

                    df['sum_pred'] = df[sub_pred_cols].sum(axis=1)

                    y_true = df[station_true_col]
                    y_pred_direct = df[station_pred_col]
                    y_pred_sum = df["sum_pred"]

                    eval_sum = evaluate(y_true, y_pred_sum)
                    eval_direct = evaluate(y_true, y_pred_direct)

                    row = {
                        'model': model_type,
                        'station': station,
                        'step': step,
                        'limit_mode': limit_mode,
                        'RMSE_sum': eval_sum['rmse'],
                        'RMSE_direct': eval_direct['rmse'],
                        'MAE_sum': eval_sum['mae'],
                        'MAE_direct': eval_direct['mae'],
                        'R2_sum': eval_sum['r2'],
                        'R2_direct': eval_direct['r2']
                    }
                    results.append(row)
                    all_results.append(row)

                    print(f"✅ 收集完成: [{model_type}] {station} | t+{step} | {limit_mode}")

        # -------------------- 保存单模型结果 --------------------
        df_result = pd.DataFrame(results)
        if df_result.empty:
            print(f"❌ [{model_type}] 没有成功收集任何结果，请检查文件结构或字段命名！")
        else:
            df_result.sort_values(['station', 'step', 'limit_mode'], inplace=True)
            out_file = f"比较分析结果_全站预测加和_vs_直接预测_{model_type}.csv"
            df_result.to_csv(out_file, index=False)
            print(f"✅ [{model_type}] 比较分析完成，结果已保存为：{out_file}")

    except Exception as e:
        import traceback
        print(f"\n❌ 模型 [{model_type}] 分析出错，已跳过。错误信息：{e}")
        traceback.print_exc()
        continue

# -------------------- 保存所有模型综合结果 --------------------
df_all_result = pd.DataFrame(all_results)
if df_all_result.empty:
    print("\n❌ 没有成功收集任何结果，请检查文件结构或字段命名！")
else:
    df_all_result.sort_values(['model', 'station', 'step', 'limit_mode'], inplace=True)
    df_all_result.to_csv("比较分析结果_全站预测加和_vs_直接预测_ALL.csv", index=False)
    print("\n🎉 所有模型分析完成！综合结果已保存为：比较分析结果_全站预测加和_vs_直接预测_ALL.csv")
