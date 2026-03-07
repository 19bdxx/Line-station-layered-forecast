import os
import pandas as pd

# -------------------- 脚本说明 --------------------
#
# 本脚本用于对比 #1开始训练_完善版.py（基准方案，每条线路独立建模）
# 与 #4跨线路特征增强训练.py（跨线路增强方案，引入同场站其他线路历史特征）
# 在相同实验设置下的预测精度差异。
#
# 【前置条件】
#   - 已运行 #1开始训练_完善版.py，生成：
#       所有实验汇总_metrics_ALL.csv
#   - 已运行 #4跨线路特征增强训练.py，生成：
#       所有实验汇总_crossline_metrics_ALL.csv
#
# 【比较维度】
#   按 (station, model, limit_mode, target, step) 进行配对比较：
#     ΔRMSE  = rmse_基准  - rmse_跨线路   （正值 = 跨线路方案 RMSE 更低，即更优）
#     ΔMAE   = mae_基准   - mae_跨线路    （同上）
#     ΔR2    = r2_跨线路  - r2_基准       （正值 = 跨线路方案 R² 更高，即更优）
#
# 【输出文件】（保存在脚本运行目录，即项目根目录）
#   对比结果_基准_vs_跨线路增强_ALL.csv          — 全量逐实验对比表
#   对比结果_基准_vs_跨线路增强_{model}.csv      — 分模型对比表
#   对比汇总_按模型_场站.csv                     — 按 (model, station) 汇总的平均改善量
#
# 【胜率定义】
#   跨线路方案在某维度的"胜率" = ΔRMSE > 0 的实验占全部配对实验的比例。
#
# ==================== 配置 ====================

BASELINE_FILE   = '所有实验汇总_metrics_ALL.csv'
CROSSLINE_FILE  = '所有实验汇总_crossline_metrics_ALL.csv'
JOIN_KEYS       = ['station', 'model', 'limit_mode', 'target', 'step']
METRICS         = ['rmse', 'mae', 'r2']

# ==================== 辅助函数 ====================

def load_and_validate(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {label} 文件：{path}\n"
            f"请先运行对应的训练脚本生成该文件。"
        )
    df = pd.read_csv(path)
    missing = [c for c in JOIN_KEYS + METRICS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} 文件缺少以下列：{missing}")
    print(f"✅ 已加载 {label}：{path}  ({len(df)} 条记录)")
    return df


def win_rate(series: pd.Series) -> float:
    """series 中正值（跨线路更优）的比例，保留 2 位小数百分比。"""
    return round((series > 0).mean() * 100, 2)


# ==================== 主流程 ====================

print("=" * 65)
print("📊 对比分析：基准方案 vs 跨线路特征增强方案")
print("=" * 65)

# ---- 1. 加载数据 ----
df_base  = load_and_validate(BASELINE_FILE,  "基准方案")
df_cross = load_and_validate(CROSSLINE_FILE, "跨线路增强方案")

# ---- 2. 重命名指标列以区分两种方案 ----
rename_base  = {m: f'{m}_基准'   for m in METRICS}
rename_cross = {m: f'{m}_跨线路' for m in METRICS}

df_base  = df_base [JOIN_KEYS + METRICS].rename(columns=rename_base)
df_cross = df_cross[JOIN_KEYS + METRICS].rename(columns=rename_cross)

# ---- 3. 配对合并（inner join，仅保留两侧均有结果的实验） ----
df = pd.merge(df_base, df_cross, on=JOIN_KEYS, how='inner')
n_base_only  = len(df_base)  - len(df)
n_cross_only = len(df_cross) - len(df)

print(f"\n配对成功：{len(df)} 个实验")
if n_base_only > 0:
    print(f"  ⚠️ 仅基准方案有结果（跨线路方案缺失）：{n_base_only} 个")
if n_cross_only > 0:
    print(f"  ⚠️ 仅跨线路方案有结果（基准缺失）：{n_cross_only} 个")

if df.empty:
    print("\n❌ 没有任何配对实验，请检查两个 CSV 文件的内容是否一致。")
    raise SystemExit(1)

# ---- 4. 计算改善量 ----
# RMSE / MAE：基准 - 跨线路，正值 = 跨线路更低（更好）
df['ΔRMSE'] = df['rmse_基准'] - df['rmse_跨线路']
df['ΔMAE']  = df['mae_基准']  - df['mae_跨线路']
# R²：跨线路 - 基准，正值 = 跨线路更高（更好）
df['ΔR2']   = df['r2_跨线路'] - df['r2_基准']

# ---- 5. 全量输出 ----
col_order = JOIN_KEYS + [
    'rmse_基准', 'rmse_跨线路', 'ΔRMSE',
    'mae_基准',  'mae_跨线路',  'ΔMAE',
    'r2_基准',   'r2_跨线路',   'ΔR2',
]
df_out = df[col_order].sort_values(JOIN_KEYS)

out_all = '对比结果_基准_vs_跨线路增强_ALL.csv'
df_out.to_csv(out_all, index=False)
print(f"\n✅ 全量对比表已保存：{out_all}")

# ---- 6. 分模型输出 + 打印汇总 ----
model_list = df_out['model'].unique()
for model in sorted(model_list):
    df_m = df_out[df_out['model'] == model]
    out_m = f'对比结果_基准_vs_跨线路增强_{model}.csv'
    df_m.to_csv(out_m, index=False)
    print(f"  ✅ [{model}] 已保存：{out_m}")

# ---- 7. 按 (model, station) 汇总平均改善量 & 胜率 ----
print("\n" + "=" * 65)
print("📈 按 (模型, 场站) 汇总：平均改善量 & RMSE 胜率")
print("=" * 65)

group_cols = ['model', 'station']
agg = df.groupby(group_cols).agg(
    实验数=('ΔRMSE', 'count'),
    ΔRMSE均值=('ΔRMSE', 'mean'),
    ΔMAE均值 =('ΔMAE',  'mean'),
    ΔR2均值  =('ΔR2',   'mean'),
    RMSE胜率_pct=('ΔRMSE', win_rate),
).reset_index()

agg = agg.sort_values(['model', 'station'])
agg['ΔRMSE均值'] = agg['ΔRMSE均值'].round(4)
agg['ΔMAE均值']  = agg['ΔMAE均值'].round(4)
agg['ΔR2均值']   = agg['ΔR2均值'].round(6)

print(agg.to_string(index=False))

out_agg = '对比汇总_按模型_场站.csv'
agg.to_csv(out_agg, index=False)
print(f"\n✅ 按 (模型, 场站) 汇总表已保存：{out_agg}")

# ---- 8. 按模型整体胜率总结 ----
print("\n" + "=" * 65)
print("🏆 按模型整体：平均 ΔRMSE & RMSE 胜率（跨线路 > 基准 的比例）")
print("=" * 65)

overall = df.groupby('model').agg(
    实验数=('ΔRMSE', 'count'),
    ΔRMSE均值=('ΔRMSE', 'mean'),
    RMSE胜率_pct=('ΔRMSE', win_rate),
    ΔR2均值  =('ΔR2',   'mean'),
).reset_index().sort_values('ΔRMSE均值', ascending=False)

overall['ΔRMSE均值'] = overall['ΔRMSE均值'].round(4)
overall['ΔR2均值']   = overall['ΔR2均值'].round(6)
print(overall.to_string(index=False))

print("\n🎉 对比分析完成！所有结果文件已保存在项目根目录。")
print("   可用文件：")
print(f"   • {out_all}")
for model in sorted(model_list):
    print(f"   • 对比结果_基准_vs_跨线路增强_{model}.csv")
print(f"   • {out_agg}")
