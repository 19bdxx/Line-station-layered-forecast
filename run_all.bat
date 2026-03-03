@echo off
chcp 65001 >nul
REM ============================================================
REM run_all.bat — 风电场功率分层预测实验一键运行脚本（Windows）
REM 使用方式：在项目根目录下双击运行，或在命令提示符中执行 run_all.bat
REM ============================================================

echo ======================================================
echo   风电场功率分层预测实验 — 一键运行
echo ======================================================
echo.

REM 步骤一：模型训练
echo [步骤 1/4] 开始模型训练... (预计 1~4 小时)
python "CODE/#1开始训练.py"
if errorlevel 1 (
    echo 错误：步骤一失败。
    echo 常见原因：依赖库未安装（运行 pip install pandas numpy lightgbm scikit-learn matplotlib）
    echo           或 RAW_DATA\all_stations.csv 文件不存在。
    pause
    exit /b 1
)
echo 步骤一完成：预测结果已保存至 对比实验\ 和 对比实验_无限电\
echo.

REM 步骤二：合并预测结果
echo [步骤 2/4] 合并预测结果...
python "CODE/#2预测结果汇总.py"
if errorlevel 1 (
    echo 错误：步骤二失败。
    echo 常见原因：步骤一未成功完成，对比实验\ 目录不存在或为空。
    pause
    exit /b 1
)
echo 步骤二完成：合并结果已保存至 合并结果\
echo.

REM 步骤三：对比分析
echo [步骤 3/4] 对比分析（分层预测加和 vs 直接预测）...
python "CODE/#3汇总结果分析.py"
if errorlevel 1 (
    echo 错误：步骤三失败。
    echo 常见原因：步骤二未成功完成，合并结果\ 目录不存在或缺少 *_merged.csv 文件。
    pause
    exit /b 1
)
echo 步骤三完成：分析结果已保存至 比较分析结果_全站预测加和_vs_直接预测.csv
echo.

REM 步骤四：可视化
echo [步骤 4/4] 生成可视化图表...
python "CODE/#4预测结果可视化.py"
if errorlevel 1 (
    echo 错误：步骤四失败。
    echo 常见原因：步骤二或步骤三未成功完成，合并结果\ 或分析 CSV 文件缺失。
    pause
    exit /b 1
)
echo 步骤四完成：图表已保存至 可视化结果\
echo.

echo ======================================================
echo   所有实验完成！
echo   主要输出文件：
echo     - 所有实验汇总_metrics.csv
echo     - 比较分析结果_全站预测加和_vs_直接预测.csv
echo     - 可视化结果\*.png
echo ======================================================
pause
