"""同一启动会话传递随机本机设置令牌；仅绑定 loopback，不输出密钥。"""
import os
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
import json
from urllib.request import urlopen
from urllib.error import URLError
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    for port in (8000,8501):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1",port))
            except OSError:
                print(f"端口 {port} 已占用，请先关闭旧服务。")
                return 1
    env=os.environ.copy()
    env["V2_SETTINGS_TOKEN"]=secrets.token_urlsafe(32)
    env["V2_API_URL"]="http://127.0.0.1:8000"
    env["PYTHONDONTWRITEBYTECODE"]="1"
    processes=[]
    print("启动 V2（本机单用户 / 单进程）；不会自动调用模型、导入文件或切换知识库。")
    print("实际向量服务与集合将在页面显示；固定 DashScope / text-embedding-v4 / v2_notes。")
    try:
        processes.append(subprocess.Popen([sys.executable,"-m","uvicorn","app.api.main:app","--host","127.0.0.1",
            "--port","8000","--no-proxy-headers","--no-access-log"],cwd=root,env=env))
        processes.append(subprocess.Popen([sys.executable,"-m","streamlit","run",str(root/"frontend/app.py"),
            "--server.address","127.0.0.1","--server.port","8501","--server.headless","true",
            "--browser.gatherUsageStats","false","--server.maxUploadSize","100"],cwd=root,env=env))
        print("前端：http://127.0.0.1:8501  |  Swagger：http://127.0.0.1:8000/docs")
        print("按 Ctrl+C 停止两个服务。")
        ready=False
        for attempt in range(120):
            if any(p.poll() is not None for p in processes):
                break
            try:
                with urlopen("http://127.0.0.1:8000/health",timeout=.5) as result:
                    health=json.load(result)["data"]
                with urlopen("http://127.0.0.1:8501/_stcore/health",timeout=.5):
                    pass
                ready=True
                break
            except (URLError,OSError,ValueError):
                time.sleep(.25)
        if not ready:
            print("服务未正常启动，请查看上方错误；未自动请求模型。")
            return 1
        print("文本模型："+health["text_model"]+"；Embedding："+health["embedding_model"]+"；Collection："+health["collection"])
        webbrowser.open("http://127.0.0.1:8501")
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
