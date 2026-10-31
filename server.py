# -*- coding: utf-8 -*-
"""
server.py — Live2D 渲染 + 情绪直驱服务

同时做两件事：
  1. 服务静态文件（脚本所在目录）
  2. 暴露 /api/emotion 端点，返回情绪引擎的 Live2D 参数

网页定时轮询 /api/emotion，把参数应用到 Live2D 模型上，
实现「情绪 → 表情」的直驱。

用法：
  python server.py [--port 8899]
"""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 引擎与静态文件根目录（脚本所在目录）
ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

try:
    from engine import EmotionEngine
except ImportError as e:  # noqa: BLE001
    print(f"无法加载情绪引擎：{e}")
    EmotionEngine = None  # type: ignore

# 模块级单例：避免每次轮询请求都重新读盘 + 解析 JSON（审查 #5）
_ENGINE_SINGLETON = None


def _get_engine():
    global _ENGINE_SINGLETON
    if EmotionEngine is None:
        return None
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = EmotionEngine()
    return _ENGINE_SINGLETON


# 允许读取 /api/emotion 的来源。默认只放行本机页面；
# 原实现无条件 Access-Control-Allow-Origin: *，任意网页都能
# fetch 本机端口读取心情。需要别的来源时用 --allow-origin 显式加。
ALLOWED_ORIGINS = {
    "http://127.0.0.1:8899", "http://localhost:8899",
    "http://127.0.0.1", "http://localhost",
}


class Handler(SimpleHTTPRequestHandler):
    """静态文件 + /api/emotion 端点。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        # 情绪参数端点
        if self.path.startswith("/api/emotion"):
            self._handle_emotion()
            return
        # 其余走静态文件
        super().do_GET()

    def _handle_emotion(self):
        """返回情绪引擎的 Live2D 参数 + 心情描述。"""
        try:
            if EmotionEngine is None:
                payload = {"error": "情绪引擎不可用"}
            else:
                eng = _get_engine()
                # 只读展示路径：轻量 reload 拿最新状态（无锁竞争，不做写事务）
                eng.state.reload()
                # mood 给渲染层的是【纯展示文本】，不再塞 emotion_context()：
                # 后者是给 LLM 的指令段落，可能含本机路径等私有信息。
                # 要完整注入文本的调用方自行调用 engine.emotion_context()。
                payload = {
                    "params": eng.live2d_params(),
                    "mood": eng.mood_text(),
                    "mood_label": eng.mood_label(),
                    "period": eng.period_text(),
                    "pad": eng.state.get_pad(),
                    "possessiveness": eng.state.get_possessiveness(),
                }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            origin = self.headers.get("Origin")
            if origin and origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        # 静默静态文件日志，只打印情绪端点
        if "/api/emotion" in (args[0] if args else ""):
            super().log_message(format, *args)


def main():
    port = 8899
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    if "--allow-origin" in sys.argv:
        origin = sys.argv[sys.argv.index("--allow-origin") + 1]
        ALLOWED_ORIGINS.add(origin)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Live2D 渲染 + 情绪直驱服务已启动：http://127.0.0.1:{port}")
    print(f"情绪端点：http://127.0.0.1:{port}/api/emotion")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
