import os
import pandas as pd

# -------------------- 脚本说明 --------------------
# 本脚本用于合并风电场功率预测结果文件，按以下方式组织和输出数据：
# 
# 【输入结构】（脚本 #1 多模型运行后自动生成）
# - 两个预测结果根目录："对比实验"（含限电）与 "对比实验_无限电"（无限电）
# - 每个根目录下按模型类型分子目录，再按场站分子目录：
#     对比实验/{model_type}/{station}/{target_col}_t+{step}.csv
# - 每个 CSV 文件包含字段：timestamp, y_true, y_pred, bias_rate, limit_value
#
# 【处理逻辑】
# - 遍历所有模型类型，对每个模型按场站、预测步长汇总合并
# - 在合并过程中对 y_true、y_pred 等字段加入目标后缀（如 JIA_true）
#
# 【输出结构】
# - 合并后文件保存在 "合并结果" 目录下：
#     合并结果/{model_type}/with_limit/{station}/{station}_t+{step}_merged.csv
#     合并结果/{model_type}/no_limit/{station}/{station}_t+{step}_merged.csv


# 自动运行的模型列表（与脚本 #1 保持一致）
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'ridge', 'mlp', 'lstm']

# 站点配置信息
stations = {
    "XYA": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI', 'XYA_ACTIVE_POWER_STATION'],
    "XYB": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU', 'XYB_ACTIVE_POWER_STATION'],
    "XS":  ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI', 'XS_ACTIVE_POWER_STATION']
}

for model_type in MODEL_TYPES:
    print(f"\n{'='*60}")
    print(f"🔄 正在合并模型：{model_type}")
    print(f"{'='*60}")

    base_dirs = {
        "with_limit": os.path.join("对比实验", model_type),
        "no_limit":   os.path.join("对比实验_无限电", model_type)
    }

    for limit_mode, base_dir in base_dirs.items():
        if not os.path.exists(base_dir):
            print(f"⚠️ 目录不存在，跳过：{base_dir}")
            continue
        print(f"\n📂 正在处理目录：{base_dir} ({limit_mode})")

        for station, targets in stations.items():
            station_dir = os.path.join(base_dir, station)
            if not os.path.exists(station_dir):
                print(f"⚠️ 目录不存在: {station_dir}")
                continue

            # 用于分步长收集合并文件
            step_grouped = {}

            for file in os.listdir(station_dir):
                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(station_dir, file)

                # 解析文件名：格式应为 {target_col}_t+{step}.csv
                try:
                    target_part, step_part = file.replace(".csv", "").split("_t+")
                    step = int(step_part)
                    target_name = target_part.split("_")[-1]  # 如 JIA, YI, STATION
                except Exception as e:
                    print(f"❌ 文件命名错误: {file} -> {e}")
                    continue

                df = pd.read_csv(file_path)
                rename_map = {
                    'y_true': f'{target_name}_true',
                    'y_pred': f'{target_name}_pred',
                    'bias_rate': f'{target_name}_bias_rate',
                    'limit_value': f'{target_name}_limit_value'
                }
                df = df.rename(columns=rename_map)

                # 保留 timestamp + 改名后的列
                use_cols = ['timestamp'] + list(rename_map.values())
                df = df[use_cols]

                if step not in step_grouped:
                    step_grouped[step] = df
                else:
                    # 合并相同 timestamp 的预测结果
                    step_grouped[step] = pd.merge(
                        step_grouped[step], df, on='timestamp', how='outer'
                    )

            # 输出每个预测步长的合并结果
            save_dir = os.path.join("合并结果", model_type, limit_mode, station)
            os.makedirs(save_dir, exist_ok=True)

            for step, df_merged in step_grouped.items():
                df_merged.sort_values("timestamp", inplace=True)
                output_path = os.path.join(save_dir, f"{station}_t+{step}_merged.csv")
                df_merged.to_csv(output_path, index=False)

            print(f"✅ [{model_type}] {station} 场站已完成合并（共 {len(step_grouped)} 个预测步长）")