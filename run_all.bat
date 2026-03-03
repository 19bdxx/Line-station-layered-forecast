@echo off
REM ============================================================
REM run_all.bat - Wind Power Forecast Pipeline (Windows)
REM Run from the project root directory: run_all.bat
REM ============================================================

echo ======================================================
echo   Wind Power Forecast - Layered Prediction Pipeline
echo ======================================================
echo.

REM Step 1: Model training (slowest step, ~1-4 hours)
echo [Step 1/4] Training models... (estimated 1-4 hours)
python "CODE/#1开始训练.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Step 1 failed.
    echo   Possible causes:
    echo     1. Missing packages - run: pip install pandas numpy lightgbm scikit-learn matplotlib
    echo     2. Data file not found: RAW_DATA\all_stations.csv
    echo.
    pause
    exit /b 1
)
echo [OK] Step 1 done. Results saved to folders: duibishi\ and duibishi_wuxiandian\
echo.

REM Step 2: Merge prediction results
echo [Step 2/4] Merging prediction results...
python "CODE/#2预测结果汇总.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Step 2 failed.
    echo   Possible cause: Step 1 did not complete successfully.
    echo   Check that the experiment output folders (duibishi\) exist and are not empty.
    echo.
    pause
    exit /b 1
)
echo [OK] Step 2 done. Merged results saved to folder: hebingjieguo\
echo.

REM Step 3: Comparative analysis
echo [Step 3/4] Running comparative analysis (layered-sum vs direct prediction)...
python "CODE/#3汇总结果分析.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Step 3 failed.
    echo   Possible cause: Step 2 did not complete successfully.
    echo   Check that merged results folder (hebingjieguo\) exists and contains *_merged.csv files.
    echo.
    pause
    exit /b 1
)
echo [OK] Step 3 done. Analysis saved to: bijiao_fenxi_jieguo.csv
echo.

REM Step 4: Visualization
echo [Step 4/4] Generating visualization charts...
python "CODE/#4预测结果可视化.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Step 4 failed.
    echo   Possible cause: Step 2 or Step 3 did not complete successfully.
    echo   Check that merged results and analysis CSV exist.
    echo.
    pause
    exit /b 1
)
echo [OK] Step 4 done. Charts saved to folder: keshihua_jieguo\
echo.

echo ======================================================
echo   All steps completed successfully!
echo   Key output files:
echo     - suoyou_shiyan_huizong_metrics.csv
echo     - bijiao_fenxi_jieguo_vs_zhijie_yuce.csv
echo     - keshihua_jieguo\*.png
echo ======================================================
pause
