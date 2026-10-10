"""本机用户凭据存储与一次服务端验证；不把明文密钥写入普通文件。"""
import ctypes
import json
import math
import os
from ctypes import wintypes
from datetime import datetime, timezone

import httpx

from ..rag.vector_retrieval import EMBEDDING_MODEL, EMBEDDING_DIMENSION

PROVIDERS = {"deepseek": "DEEPSEEK_API_KEY", "dashscope": "DASHSCOPE_API_KEY"}


class CredentialRecord:
    def __init__(self, secret, metadata=None):
        self.secret, self.metadata = secret, metadata or {}


class WindowsCredentials:
    """Windows Generic Credential；仅当前 Windows 用户可读取，重启后保留。"""
    def __init__(self, prefix="PersonalNotesReviewAssistant/V2"):
        if os.name != "nt":
            raise ValueError("当前环境不支持 Windows 系统凭据。")
        self.prefix = prefix
        class Credential(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                        ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
                        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                        ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]
        self.structure = Credential
        self.api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        self.api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.POINTER(ctypes.POINTER(Credential))]
        self.api.CredReadW.restype = wintypes.BOOL
        self.api.CredWriteW.argtypes = [ctypes.POINTER(Credential), wintypes.DWORD]
        self.api.CredWriteW.restype = wintypes.BOOL
        self.api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.api.CredDeleteW.restype = wintypes.BOOL
        self.api.CredFree.argtypes = [ctypes.c_void_p]
        self.api.CredFree.restype = None

    def read(self, name):
        pointer = ctypes.POINTER(self.structure)()
        if not self.api.CredReadW(self.prefix + "/" + name, 1, 0, ctypes.byref(pointer)):
            if ctypes.get_last_error() == 1168:
                return None
            raise ValueError("系统凭据读取失败。")
        try:
            record = pointer.contents
            secret = ctypes.string_at(record.CredentialBlob, record.CredentialBlobSize).decode("utf-8")
            return CredentialRecord(secret, json.loads(record.Comment or "{}"))
        finally:
            self.api.CredFree(pointer)

    def write(self, name, secret, metadata):
        value = secret.encode("utf-8")
        if not value or len(value)>2560:
            raise ValueError("凭据长度不符合系统限制。")
        blob = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        record = self.structure(Type=1, TargetName=self.prefix + "/" + name,
                                 Comment=json.dumps(metadata, ensure_ascii=False),
                                 CredentialBlobSize=len(value), CredentialBlob=blob, Persist=2,
                                 UserName="V2 local user")
        if not self.api.CredWriteW(ctypes.byref(record), 0):
            raise ValueError("系统凭据保存失败。")

    def delete(self, name):
        if not self.api.CredDeleteW(self.prefix + "/" + name, 1, 0) and ctypes.get_last_error()!=1168:
            raise ValueError("系统凭据删除失败。")


def verify_key(provider, secret, *, transport=None):
    """【业务需求】保存前进行一次真实服务端校验，不以长度或前缀判有效。
    【输入】provider、候选密钥；测试可注入 Mock HTTP。
    【执行步骤】固定端点发出一次无重试请求，验证实际业务响应；不写 Chroma。
    【输出】非敏感验证能力/请求数；失败抛安全错误，不能覆盖旧配置。
    """
    if provider == "dashscope":
        # 原生文本 Embedding 协议；短文本只发一次请求，无 SDK 自动重试。
        try:
            with httpx.Client(timeout=30, trust_env=False, follow_redirects=False,
                    transport=transport or httpx.HTTPTransport(retries=0)) as client:
                response = client.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    headers={"Authorization": "Bearer " + secret},
                    json={"model": EMBEDDING_MODEL, "input": {"texts": ["本机服务配置验证"]},
                          "parameters": {"text_type": "query", "dimension": EMBEDDING_DIMENSION,
                                         "output_type": "dense"}})
                response.raise_for_status()
                rows = response.json()["output"]["embeddings"]
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError("向量数量错误")
                vector = rows[0]["embedding"]
                if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION or any(
                        isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) for value in vector):
                    raise ValueError("向量格式错误")
        except Exception:
            raise ValueError("验证失败，请检查密钥、服务权限或网络；原配置已保留。") from None
        return {"capabilities": ["text_embedding_1024"], "model": EMBEDDING_MODEL,
                "http_requests": 1}
    if provider != "deepseek":
        raise ValueError("不支持的服务。")
    model = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")
    with httpx.Client(base_url="https://api.deepseek.com", timeout=30,
                      headers={"Authorization": "Bearer " + secret}, trust_env=False,
                      follow_redirects=False, transport=transport or httpx.HTTPTransport(retries=0)) as client:
        response = client.post("/chat/completions", json={"model":model,"max_tokens":8,"stream":False,
            "thinking":{"type":"disabled"},"messages":[{"role":"user","content":"回复 OK"}]})
        try:
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("空响应")
        except Exception:
            raise ValueError("验证失败，请检查密钥、服务权限或网络；原配置已保留。") from None
    return {"capabilities":["text_chat"],"model":model,"http_requests":1}


class KeyManager:
    def __init__(self, storage, *, environment=None, validator=verify_key, storage_source="windows_credentials"):
        self.storage = storage
        self.storage_source = storage_source
        self.environment = os.environ if environment is None else environment
        self.validator = validator

    def effective(self, provider):
        self._check_provider(provider)
        if provider == "deepseek" and self.storage and self.storage.read("disabled/" + provider):
            return ""
        saved = self.storage.read(provider) if self.storage else None
        return saved.secret if saved else self.environment.get(PROVIDERS[provider], "")

    def status(self, provider):
        self._check_provider(provider)
        disabled = provider == "deepseek" and self.storage and self.storage.read("disabled/" + provider)
        record = self.storage.read(provider) if self.storage and not disabled else None
        secret = "" if disabled else record.secret if record else self.environment.get(PROVIDERS[provider], "")
        return {"provider":provider,"configured":bool(secret),
                "masked_key": "••••" + secret[-4:] if len(secret)>=12 else "已配置" if secret else "",
                "source":"disabled" if disabled else self.storage_source if record else "environment" if secret else "none",
                "validation_state":"verified" if record else "not_verified" if secret else "not_configured",
                "capabilities":record.metadata.get("capabilities",[]) if record else [],
                "validated_at":record.metadata.get("validated_at") if record else None}

    def save(self, provider, secret):
        """【业务需求】验证成功后才替换凭据，失败保留原有效 Key。
        【输入】服务与候选 Key。
        【执行步骤】验证一次；写系统凭据；解除禁用标记；持久化异常回滚。
        【输出】脱敏配置状态，绝不返回完整 Key。
        """
        self._check_provider(provider)
        verified = self.validator(provider, secret)
        metadata = {**verified,"validated_at":datetime.now(timezone.utc).isoformat()}
        old = self.storage.read(provider)
        try:
            self.storage.write(provider, secret, metadata)
            self.storage.delete("disabled/" + provider)
        except Exception:
            if old:
                self.storage.write(provider, old.secret, old.metadata)
            else:
                self.storage.delete(provider)
            raise ValueError("安全保存失败，未启用新配置。") from None
        return self.status(provider)

    @staticmethod
    def _check_provider(provider):
        if provider not in PROVIDERS:
            raise ValueError("不支持的服务。")

    def delete(self, provider):
        self._check_provider(provider)
        if provider == "dashscope":
            # 用户删除保存配置后，明确恢复环境配置；不修改 .env。
            self.storage.delete(provider)
            self.storage.delete("disabled/" + provider)
            return self.status(provider)
        # 先持久化禁用状态，避免删除后悄悄使用 .env 的另一把 Key。
        self.storage.write("disabled/" + provider, "disabled", {})
        self.storage.delete(provider)
        return self.status(provider)
