"""只在云端容器内部监听，Streamlit 重跑不重复启动后端。"""
import atexit
import socket
import threading
import time

import uvicorn

from app.api.main import create_app
from app.api.cloud import CloudSessions, CloudSessionGuard


def start_backend(sessions=None):
    """【业务需求】保留 HTTP/SSE 真流，同进程只启动一个内部服务。
    【执行步骤】创建隔离注册表；绑定随机 loopback 端口；后台运行并确认就绪。
    【输出】注册表和内部 URL；启动失败明确抛错，不请求供应商。
    """
    sessions = sessions if sessions is not None else CloudSessions()
    app = create_app()
    app.add_middleware(CloudSessionGuard, sessions=sessions)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
        log_level="critical", access_log=False, proxy_headers=False))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    sessions.server = server
    sessions.server_thread = thread
    deadline = time.monotonic() + 20
    while thread.is_alive() and not server.started and time.monotonic() < deadline:
        time.sleep(.05)
    if not server.started:
        server.should_exit = True
        sock.close()
        raise RuntimeError("内部后端启动失败。")
    def cleanup():
        while not server.should_exit:
            time.sleep(30)
            sessions.prune()
    threading.Thread(target=cleanup, daemon=True).start()
    atexit.register(lambda: setattr(server, "should_exit", True))
    return sessions, "http://127.0.0.1:" + str(port)
