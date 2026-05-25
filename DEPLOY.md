# 公网分享部署指南

本项目使用 Streamlit 构建前端，支持通过 ngrok 内网穿透实现公网远程访问。

## 前提条件

1. 已安装 Python 3.11+
2. 已注册 ngrok 账号（https://ngrok.com，免费）

## 快速启动（局域网）

双击 `start_server.bat`，等待出现启动信息后访问 http://localhost:8501

## 公网分享步骤

### 1. 安装 ngrok

- 访问 https://ngrok.com/download 下载 Windows 版
- 解压到任意目录（如 `C:\ngrok\`）
- 注册账号后，在 https://dashboard.ngrok.com/get-started/setup 获取 authtoken
- 在命令行运行：

```bash
ngrok config add-authtoken 你的token
```

### 2. 启动 Streamlit

双击 `start_server.bat`，等待出现：

```
You can now view your Streamlit app in your browser
```

### 3. 启动 ngrok 隧道

打开**另一个**命令行窗口，运行：

```bash
ngrok http 8501
```

在输出中找到 `Forwarding` 行：

```
Forwarding  https://xxxx-xxx.ngrok-free.app -> http://localhost:8501
```

将 `https://xxxx-xxx.ngrok-free.app` 这个链接发给别人即可访问。

### 4. 停止服务

关闭两个命令行窗口即可。

## 注意事项

- ngrok 免费版链接**每次启动会变化**，这是正常的
- 电脑需要保持开机状态，关闭终端则服务停止
- RAG 诊断引擎暂未部署，仅展示模式（看板 + 深度诊断详情页可用，实时重算和问答功能暂不可用）
- 未来部署 RAG 到云服务器后，可通过 FastAPI 接口接入，无需改动前端

## 手动启动（不用 bat）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501
```
