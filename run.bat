@echo off
REM ============================================================
REM  RansomGuard AI · Windows 一键启动脚本
REM  - 自动检测 Python
REM  - 自动安装依赖 (仅首次较慢)
REM  - 启动 Flask Web UI
REM  - 自动用默认浏览器打开 http://127.0.0.1:5000
REM ============================================================

REM 设置 UTF-8 输出编码 (避免中文乱码)
chcp 65001 >nul 2>&1

REM 切换到脚本所在目录 (支持任意路径双击运行)
cd /d "%~dp0"

echo ============================================
echo  RansomGuard AI - 勒索软件检测系统
echo ============================================
echo.

REM 检测 Python
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未检测到 Python!
  echo 请先访问 https://www.python.org/downloads/ 安装 Python 3.9 或更高版本。
  echo 安装时请务必勾选 "Add Python to PATH"。
  pause
  exit /b 1
)

echo [步骤 1/3] 检查依赖...
python -c "import flask, pandas, numpy, sklearn, requests, pefile" 2>nul
if errorlevel 1 (
  echo   依赖未完全安装，正在运行 pip install ...
  echo   (首次启动可能需要 2-5 分钟，请耐心等待)
  echo.
  python -m pip install flask pandas numpy scikit-learn requests pefile
  if errorlevel 1 (
    echo.
    echo   [!] pip install 返回非零退出代码，尝试继续启动...
  )
) else (
  echo   所有依赖已就绪 - OK
)

echo.
echo [步骤 2/3] 检查示例数据...
if not exist "data\sample_confidence.csv" (
  echo   首次启动，正在生成示例数据...
  python main.py init-data
) else (
  echo   示例数据已就绪 - OK
)

echo.
echo [步骤 3/3] 启动 Web UI...
echo   服务地址: http://127.0.0.1:5000
echo   如浏览器将在 4 秒后自动打开
echo   如需停止服务: 直接关闭此窗口 或 按 Ctrl+C
echo.

REM 启动一个后台任务: 等待 4 秒让 Flask 启动完成, 然后自动打开浏览器
start "" cmd /c "timeout /t 4 /nobreak >nul 2>&1 & start http://127.0.0.1:5000"

REM 启动 Flask 服务器 (前台)
python main.py serve --host 127.0.0.1 --port 5000

echo.
echo 服务器已停止。
pause