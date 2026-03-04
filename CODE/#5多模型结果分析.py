import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -------------------- 代码说明 --------------------
# 本脚本读取 #4多模型对比训练.py 生成的多模型预测结果，执行以下分析：
#
# 1. 合并（Merge）：对每个模型 × 场站 × 限电模式，按预测步长将各目标列
#    预测结果 CSV 横向合并（逻辑与脚本#2相同）。
#
# 2. 对比（Compare）：对每个合并结果，计算并对比两种全站预测策略：
#    - 直接预测（DIRECT）：使用 STATION_pred 列
#    - 分层聚合（SUM）  ：将各线路 *_pred 列加和
#
# 3. 跨模型汇总：统计每个模型在不同场站/步长/限电条件下，
#    SUM 优于 DIRECT（RMSE_sum < RMSE_direct）的比例，
#    判断结论一致性。
#
# 输出文件：
#   多模型对比_分层vs直接_详细结果.csv  —— 每条记录对应一个（模型,场站,步长,限电）
#   多模型对比_结论一致性汇总.csv        —— 每个模型对应"SUM获胜比例"
#   多模型对比_RMSE对比图/               —— 折线图（按场站 × 限电模式 × 模型）


# -------------------- 配置 --------------------
MODEL_LIST = ['lgb', 'rf', 'xgb', 'ridge', 'et']

stations = {
    "XYA": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI'],
    "XYB": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU'],
    "XS":  ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI']
}

BASE_RESULT_DIR = '多模型实验'
FIGURE_DIR = '多模型对比_RMSE对比图'
os.makedirs(FIGURE_DIR, exist_ok=True)


# -------------------- 工具函数 --------------------
def evaluate(y_true, y_pred):
    """计算 RMSE、MAE、R²。"""
    return {
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def merge_station_results(station_dir, station):
    """
    将场站目录下所有 {target_col}_t+{step}.csv 文件，
    按预测步长横向合并（与脚本#2逻辑相同）。

    返回 dict: {step -> merged_DataFrame}
    """
    step_grouped = {}

    if not os.path.exists(station_dir):
        return step_grouped

    for file in os.listdir(station_dir):
        if not file.endswith('.csv'):
            continue
        try:
            target_part, step_part = file.replace('.csv', '').split('_t+')
            step = int(step_part)
            target_name = target_part.split('_')[-1]  # 如 JIA, YI, STATION
        except Exception as e:
            print(f"  ⚠️ 文件命名异常: {file} -> {e}")
            continue

        df = pd.read_csv(os.path.join(station_dir, file))
        rename_map = {
            'y_true': f'{target_name}_true',
            'y_pred': f'{target_name}_pred',
            'bias_rate': f'{target_name}_bias_rate',
            'limit_value': f'{target_name}_limit_value'
        }
        df = df.rename(columns=rename_map)
        use_cols = ['timestamp'] + [v for v in rename_map.values() if v in df.columns]
        df = df[use_cols]

        if step not in step_grouped:
            step_grouped[step] = df
        else:
            step_grouped[step] = pd.merge(step_grouped[step], df, on='timestamp', how='outer')

    return step_grouped


# -------------------- 主循环 --------------------
all_results = []

for model_type in MODEL_LIST:
    model_dir = os.path.join(BASE_RESULT_DIR, model_type)
    if not os.path.exists(model_dir):
        print(f"⚠️ 模型目录不存在，跳过: {model_dir}（请先运行 #4多模型对比训练.py）")
        continue

    print(f"\n{'='*55}")
    print(f"📂 正在分析模型：{model_type.upper()}")
    print(f"{'='*55}")

    limit_dirs = {
        'with_limit': '对比实验',
        'no_limit': '对比实验_无限电'
    }

    for limit_mode, sub_dir_name in limit_dirs.items():
        for station in stations:
            station_dir = os.path.join(model_dir, sub_dir_name, station)
            step_grouped = merge_station_results(station_dir, station)

            if not step_grouped:
                continue

            for step, df_merged in step_grouped.items():
                df_merged.sort_values('timestamp', inplace=True)

                station_true_col = 'STATION_true'
                station_pred_col = 'STATION_pred'

                if station_pred_col not in df_merged.columns or station_true_col not in df_merged.columns:
                    print(f"  ⚠️ 缺少全站字段，跳过: {model_type} {station} t+{step}")
                    continue

                sub_pred_cols = [
                    c for c in df_merged.columns
                    if c.endswith('_pred') and 'STATION' not in c
                ]
                if not sub_pred_cols:
                    print(f"  ⚠️ 没有子线路预测列，跳过: {model_type} {station} t+{step}")
                    continue

                df_merged['sum_pred'] = df_merged[sub_pred_cols].sum(axis=1)
                df_valid = df_merged.dropna(subset=[station_true_col, station_pred_col, 'sum_pred'])

                y_true = df_valid[station_true_col].values
                y_direct = df_valid[station_pred_col].values
                y_sum = df_valid['sum_pred'].values

                eval_direct = evaluate(y_true, y_direct)
                eval_sum = evaluate(y_true, y_sum)
                sum_wins = int(eval_sum['rmse'] < eval_direct['rmse'])

                record = {
                    'model': model_type,
                    'station': station,
                    'step': step,
                    'limit_mode': limit_mode,
                    'RMSE_direct': round(eval_direct['rmse'], 4),
                    'RMSE_sum': round(eval_sum['rmse'], 4),
                    'MAE_direct': round(eval_direct['mae'], 4),
                    'MAE_sum': round(eval_sum['mae'], 4),
                    'R2_direct': round(eval_direct['r2'], 4),
                    'R2_sum': round(eval_sum['r2'], 4),
                    'sum_wins_rmse': sum_wins
                }
                all_results.append(record)
                print(f"  ✅ {station} t+{step:>3} | RMSE直接={eval_direct['rmse']:.2f}  RMSE加和={eval_sum['rmse']:.2f}  {'✔ 加和胜' if sum_wins else '✘ 直接胜'}")

# -------------------- 保存详细结果 --------------------
if not all_results:
    print("\n❌ 未收集到任何结果，请确认已运行 #4多模型对比训练.py 并检查目录结构。")
else:
    df_detail = pd.DataFrame(all_results)
    df_detail.sort_values(['model', 'station', 'limit_mode', 'step'], inplace=True)
    df_detail.to_csv('多模型对比_分层vs直接_详细结果.csv', index=False)
    print(f"\n📄 详细结果已保存：多模型对比_分层vs直接_详细结果.csv（共 {len(df_detail)} 条）")

    # -------------------- 结论一致性汇总 --------------------
    consistency_rows = []
    for model_type in df_detail['model'].unique():
        sub = df_detail[df_detail['model'] == model_type]
        n_total = len(sub)
        n_sum_wins = sub['sum_wins_rmse'].sum()
        win_rate = n_sum_wins / n_total if n_total > 0 else float('nan')
        consistency_rows.append({
            'model': model_type,
            'total_cases': n_total,
            'sum_wins_count': int(n_sum_wins),
            'sum_win_rate': round(win_rate, 4),
            'conclusion': '✔ 加和策略更优（与LightGBM结论一致）' if win_rate >= 0.5 else '✘ 直接策略更优（与LightGBM结论不一致）'
        })

    df_consistency = pd.DataFrame(consistency_rows)
    df_consistency.to_csv('多模型对比_结论一致性汇总.csv', index=False)

    print("\n📊 跨模型结论一致性汇总：")
    print(df_consistency[['model', 'total_cases', 'sum_wins_count', 'sum_win_rate', 'conclusion']].to_string(index=False))
    print("\n✅ 汇总已保存：多模型对比_结论一致性汇总.csv")

    # -------------------- 可视化：各场站 RMSE 随预测步长变化 --------------------
    for station in df_detail['station'].unique():
        for limit_mode in df_detail['limit_mode'].unique():
            subset = df_detail[
                (df_detail['station'] == station) &
                (df_detail['limit_mode'] == limit_mode)
            ].copy()

            if subset.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{station} | {'含限电' if limit_mode == 'with_limit' else '无限电'} — RMSE vs 预测步长", fontsize=13)

            for model_type in subset['model'].unique():
                m_data = subset[subset['model'] == model_type].sort_values('step')
                steps = m_data['step'].values
                axes[0].plot(steps, m_data['RMSE_direct'].values, marker='o', label=model_type)
                axes[1].plot(steps, m_data['RMSE_sum'].values, marker='s', label=model_type)

            for ax, title in zip(axes, ['直接预测 RMSE', '分层加和预测 RMSE']):
                ax.set_xlabel('预测步长（分钟）')
                ax.set_ylabel('RMSE')
                ax.set_title(title)
                ax.legend(fontsize=8)
                ax.grid(True)

            plt.tight_layout()
            fig_path = os.path.join(FIGURE_DIR, f"{station}_{limit_mode}_RMSE对比.png")
            plt.savefig(fig_path, dpi=120)
            plt.close()
            print(f"  📈 图像已保存：{fig_path}")

    print("\n🎉 多模型结果分析全部完成！")
