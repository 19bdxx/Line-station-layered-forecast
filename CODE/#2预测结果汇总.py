import os
import pandas as pd

# -------------------- 脚本说明 --------------------
# 本脚本用于合并风电场功率预测结果文件，按以下方式组织和输出数据：
# 
# 【输入结构】
# - 两个预测结果文件夹："对比实验"（含限电）与 "对比实验_无限电"（无限电）
# - 每个文件夹下包含三个场站目录：XYA、XYB、XS
# - 每个场站目录下包含多个 CSV 文件，命名格式为：{target_col}_t+{step}.csv
# - 每个 CSV 文件表示一个目标字段在某个预测步长下的预测结果，包含字段：
#   timestamp, y_true, y_pred, bias_rate, limit_value
#
# 【处理逻辑】
# - 针对每个场站（XYA、XYB、XS），收集该场站下所有目标字段在所有预测步长下的 CSV 文件
# - 按预测步长进行分组，将相同步长但不同目标字段的结果文件合并为一个文件
# - 在合并过程中，为避免列名冲突，对 y_true、y_pred 等字段进行重命名，加入目标后缀（如 JIA_y_true）
#
# 【输出结构】
# - 合并后的文件保存在 "合并结果" 目录下，子目录结构为：
#     合并结果/
#     ├─ with_limit/XYA/XYA_t+1_merged.csv
#     └─ no_limit/XYB/XYB_t+5_merged.csv
# - 每个合并 CSV 包含 timestamp 及所有目标字段在对应步长下的预测结果，便于后续对比分析


# 站点配置信息
stations = {
    "XYA": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI', 'XYA_ACTIVE_POWER_STATION'],
    "XYB": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU', 'XYB_ACTIVE_POWER_STATION'],
    "XS":  ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI', 'XS_ACTIVE_POWER_STATION']
}

# 目录遍历
base_dirs = {
    "with_limit": "对比实验",
    "no_limit": "对比实验_无限电"
}

for limit_mode, base_dir in base_dirs.items():
    print(f"\n{'='*50}")
    print(f"📂 处理目录：{base_dir} ({limit_mode})")
    print(f"{'='*50}")
    
    for station, targets in stations.items():
        station_dir = os.path.join(base_dir, station)
        if not os.path.exists(station_dir):
            print(f"⚠️ 目录不存在: {station_dir}")
            continue

        csv_files = [f for f in os.listdir(station_dir) if f.endswith(".csv")]
        print(f"  📁 {station}：发现 {len(csv_files)} 个 CSV 文件", flush=True)

        # 用于分步长收集合并文件
        step_grouped = {}

        for file_idx, file in enumerate(csv_files, 1):
            print(f"    [{file_idx}/{len(csv_files)}] 读取: {file}", flush=True)
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
        save_dir = os.path.join("合并结果", limit_mode, station)
        os.makedirs(save_dir, exist_ok=True)

        for step, df_merged in step_grouped.items():
            df_merged.sort_values("timestamp", inplace=True)
            output_path = os.path.join(save_dir, f"{station}_t+{step}_merged.csv")
            df_merged.to_csv(output_path, index=False)

        print(f"✅ {station} 场站已完成合并（共 {len(step_grouped)} 个预测步长）")