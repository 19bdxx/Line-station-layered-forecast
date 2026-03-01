import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------- 脚本说明 --------------------
# 本脚本对预测结果进行可视化分析，包含两类图表：
#
# 【图表一：RMSE 随预测步长变化曲线】
# - 数据来源：比较分析结果_全站预测加和_vs_直接预测.csv
# - 对每个场站分别绘图，比较：
#   - 含限电 / 无限电
#   - 分层加和预测 / 全站直接预测
# - 输出路径：可视化结果/{station}_RMSE_vs_horizon.png
#
# 【图表二：实测 vs 预测时序曲线】
# - 数据来源：合并结果/{limit_mode}/{station}/*_merged.csv
# - 对指定场站和预测步长绘制全站实测值与两种预测方法的对比曲线
# - 默认展示 t+15 和 t+60 步长，取测试集最后 N 个时间点
# - 输出路径：可视化结果/{station}_{limit_mode}_t+{step}_timeseries.png

# -------------------- 配置 --------------------
METRICS_FILE = "比较分析结果_全站预测加和_vs_直接预测.csv"
MERGED_DIR = "合并结果"
OUTPUT_DIR = "可视化结果"
PLOT_STEPS = [15, 60]    # 时序曲线展示的预测步长（分钟）
PLOT_N_POINTS = 500      # 时序曲线展示的时间点数量

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 图表一：RMSE vs 预测步长 ====================
print("📊 正在生成 RMSE vs 预测步长曲线...")

if not os.path.exists(METRICS_FILE):
    print(f"⚠️ 找不到指标汇总文件：{METRICS_FILE}，跳过图表一")
else:
    df_metrics = pd.read_csv(METRICS_FILE)
    stations = df_metrics['station'].unique()

    for station in stations:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(f"{station} 全站功率预测 RMSE 对比（分层加和 vs 直接预测）", fontsize=13)

        for ax, limit_mode in zip(axes, ['with_limit', 'no_limit']):
            sub = df_metrics[
                (df_metrics['station'] == station) &
                (df_metrics['limit_mode'] == limit_mode)
            ].sort_values('step')

            if sub.empty:
                ax.set_title(f"{'含限电' if limit_mode == 'with_limit' else '无限电'}（无数据）")
                continue

            ax.plot(sub['step'], sub['RMSE_sum'], marker='o', label='分层加和预测', linewidth=1.5)
            ax.plot(sub['step'], sub['RMSE_direct'], marker='s', linestyle='--', label='直接预测', linewidth=1.5)
            ax.set_title('含限电' if limit_mode == 'with_limit' else '无限电')
            ax.set_xlabel('预测步长（分钟）')
            ax.set_ylabel('RMSE (MW)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks(sub['step'].tolist())

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{station}_RMSE_vs_horizon.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ✅ 已保存：{save_path}")

# ==================== 图表二：实测 vs 预测时序曲线 ====================
print("\n📈 正在生成实测 vs 预测时序曲线...")

limit_modes = {
    "with_limit": "含限电",
    "no_limit": "无限电"
}

for limit_mode, limit_label in limit_modes.items():
    mode_dir = os.path.join(MERGED_DIR, limit_mode)
    if not os.path.exists(mode_dir):
        print(f"  ⚠️ 目录不存在：{mode_dir}")
        continue

    for station in os.listdir(mode_dir):
        station_dir = os.path.join(mode_dir, station)
        if not os.path.isdir(station_dir):
            continue

        for step in PLOT_STEPS:
            fname = f"{station}_t+{step}_merged.csv"
            fpath = os.path.join(station_dir, fname)
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath, parse_dates=['timestamp'])

            station_true_col = "STATION_true"
            station_pred_col = "STATION_pred"
            sub_pred_cols = [c for c in df.columns if c.endswith("_pred") and "STATION" not in c]

            if station_true_col not in df.columns or station_pred_col not in df.columns:
                print(f"  ⚠️ 缺少全站字段，跳过：{fname}")
                continue

            df = df.dropna(subset=[station_true_col, station_pred_col]).sort_values('timestamp')

            # 取最后 PLOT_N_POINTS 个点绘图（代表测试集区域）
            plot_df = df.tail(PLOT_N_POINTS)
            df['sum_pred'] = df[sub_pred_cols].sum(axis=1) if sub_pred_cols else df[station_pred_col]
            plot_df = df.tail(PLOT_N_POINTS)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(plot_df['timestamp'], plot_df[station_true_col],
                    label='实测值', linewidth=1.2, color='black')
            ax.plot(plot_df['timestamp'], plot_df[station_pred_col],
                    label='直接预测', linewidth=1.0, linestyle='--', color='steelblue')
            if sub_pred_cols:
                ax.plot(plot_df['timestamp'], plot_df['sum_pred'],
                        label='分层加和预测', linewidth=1.0, linestyle=':', color='tomato')

            ax.set_title(f"{station}（{limit_label}）全站功率 t+{step}min 预测对比（最后{PLOT_N_POINTS}个时间点）")
            ax.set_xlabel('时间')
            ax.set_ylabel('功率 (MW)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

            save_path = os.path.join(OUTPUT_DIR, f"{station}_{limit_mode}_t+{step}_timeseries.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  ✅ 已保存：{save_path}")

print(f"\n🎉 所有可视化图表已保存至：{OUTPUT_DIR}/")
