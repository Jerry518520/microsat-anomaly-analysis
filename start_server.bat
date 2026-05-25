@echo off
chcp 65001 >nul
title OPS-SAT 遥测异常诊断系统

echo ========================================
echo   OPS-SAT 遥测异常诊断系统 - 启动器
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

:: 检查 streamlit 是否已安装
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行，正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动执行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

:: 检查 CUDA
echo [检查] 验证 CUDA 环境...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>nul
if errorlevel 1 (
    echo [警告] CUDA 不可用！RAG 诊断引擎将无法启动。
    echo [提示] 请确认已安装 CUDA Toolkit 和 faiss-gpu
    echo.
)

echo [启动] 正在启动 Streamlit 服务...
echo [地址] 本机访问: http://localhost:8501
echo [提示] 如需公网分享，请在另一个窗口运行: ngrok http 8501
echo.

streamlit run src/ui/app.py --server.port 8501

pause
