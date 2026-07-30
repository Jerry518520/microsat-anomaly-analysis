"""FastAPI 入口 — 封装现有 Python 逻辑为 REST API"""
import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import dashboard, detection, explanation

app = FastAPI(
    title="OPS-SAT Telemetry API",
    description="卫星遥测异常诊断系统 REST API",
    version="2.0.0",
)

# CORS — 允许 React dev server 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])
app.include_router(explanation.router, prefix="/api/explanation", tags=["Explanation"])


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
