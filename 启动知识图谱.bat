@echo off
chcp 65001 >nul 2>&1
title 知识图谱仪表盘

set "ROOT=%~dp0"
set "NODE=%ROOT%node-runtime\node.exe"

echo ========================================
echo   微小卫星项目 - 知识图谱仪表盘
echo ========================================
echo.

if not exist "%NODE%" (
    echo [错误] 未找到内置 Node.js，确保 node-runtime 文件夹存在
    pause
    exit /b 1
)

if not exist "%ROOT%.understand-anything\knowledge-graph.json" (
    echo [错误] 未找到知识图谱，请先运行 /understand
    pause
    exit /b 1
)

set "DASHBOARD="
if exist "%USERPROFILE%\.claude\skills\Understand-Anything\understand-anything-plugin\packages\dashboard\package.json" (
    set "DASHBOARD=%USERPROFILE%\.claude\skills\Understand-Anything\understand-anything-plugin\packages\dashboard"
) else if exist "%USERPROFILE%\.agents\skills\Understand-Anything\understand-anything-plugin\packages\dashboard\package.json" (
    set "DASHBOARD=%USERPROFILE%\.agents\skills\Understand-Anything\understand-anything-plugin\packages\dashboard"
)

if "%DASHBOARD%"=="" (
    echo [错误] 未找到 dashboard 插件
    pause
    exit /b 1
)

echo 启动中，浏览器将自动打开...
echo.

cd /d "%DASHBOARD%"
set "GRAPH_DIR=%ROOT%"
"%NODE%" "node_modules\vite\bin\vite.js" --host 127.0.0.1 --open
