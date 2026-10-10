"""DeepSeek 官方只读余额接口；仅由用户主动刷新触发。"""
from datetime import datetime, timezone
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class BalanceInfo(BaseModel):
    model_config = ConfigDict(strict=True)
    currency: Literal["CNY", "USD"]
    total_balance: str = Field(pattern=r"^-?\d+(\.\d+)?$")
    granted_balance: str = Field(pattern=r"^-?\d+(\.\d+)?$")
    topped_up_balance: str = Field(pattern=r"^-?\d+(\.\d+)?$")


class BalanceResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    is_available: bool
    balance_infos: list[BalanceInfo] = Field(min_length=1)


def fetch_balance(secret, *, transport=None):
    """【业务需求】读取真实服务余额，不调用生成模型、不提供演示金额。
    【输入】后端有效密钥；离线测试可注入 Mock HTTP。
    【执行步骤】固定官方端点发出一次 GET；禁重试和重定向；校验金额与币种。
    【输出】余额、查询时间与 HTTP 状态；异常不携带 Key 或远端错误正文。
    """
    if not secret:
        raise ValueError("尚未配置 DeepSeek。")
    try:
        with httpx.Client(timeout=15, trust_env=False, follow_redirects=False,
                          transport=transport or httpx.HTTPTransport(retries=0)) as client:
            response = client.get("https://api.deepseek.com/user/balance",
                                  headers={"Authorization": "Bearer " + secret})
            response.raise_for_status()
            data = BalanceResponse.model_validate(response.json()).model_dump()
        return {**data, "queried_at": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code, "http_requests": 1}
    except Exception:
        raise ValueError("余额查询失败，请检查配置、权限或网络；未自动重试。") from None
