# 微小卫星项目 - 知识图谱仪表盘启动脚本
# 双击或右键 -> 使用 PowerShell 运行

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeExe = Join-Path $root "node-runtime\node.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  微小卫星项目 - 知识图谱仪表盘" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $nodeExe)) {
    Write-Host "[错误] 未找到内置 Node.js" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

$kgPath = Join-Path $root ".understand-anything\knowledge-graph.json"
if (-not (Test-Path $kgPath)) {
    Write-Host "[错误] 未找到知识图谱，请先运行 /understand" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

$dashboardPaths = @(
    "$env:USERPROFILE\.claude\skills\Understand-Anything\understand-anything-plugin\packages\dashboard",
    "$env:USERPROFILE\.agents\skills\Understand-Anything\understand-anything-plugin\packages\dashboard"
)

$dashboard = $null
foreach ($p in $dashboardPaths) {
    if (Test-Path "$p\package.json") {
        $dashboard = $p
        break
    }
}

if (-not $dashboard) {
    Write-Host "[错误] 未找到 dashboard 插件" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

Write-Host "启动中，浏览器将自动打开..." -ForegroundColor Green
Write-Host ""

$env:GRAPH_DIR = $root
Set-Location $dashboard
& $nodeExe "node_modules\vite\bin\vite.js" --host 127.0.0.1 --open
