"""余额只读接口离线验证：所有供应商请求经过 Mock HTTP。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from fastapi.testclient import TestClient
from app.api.balance import fetch_balance
from app.api.credentials import KeyManager
from app.api.main import create_app
from app.api.runtime import Runtime
from smoke_frontend_settings import MemoryCredentials

SECRET="offline-balance-secret"


def payload():
    return {"is_available":True,"balance_infos":[{"currency":"CNY","total_balance":"12.50",
            "granted_balance":"2.50","topped_up_balance":"10.00"}]}


class BalanceTests(unittest.TestCase):
    def setUp(self):
        self.storage=MemoryCredentials()
        self.keys=KeyManager(self.storage,environment={"DEEPSEEK_API_KEY":SECRET})
        self.calls=[]
        self.reply=httpx.Response(200,json=payload())
        def handler(request):
            self.calls.append(request)
            return self.reply
        self.reader=lambda key:fetch_balance(key,transport=httpx.MockTransport(handler))
        self.runtime=Runtime(keys=self.keys,settings_token="offline-token",balance_reader=self.reader)
        self.client=TestClient(create_app(self.runtime),base_url="http://127.0.0.1",client=("127.0.0.1",1))
        self.addCleanup(self.client.close)
        self.headers={"X-V2-Settings-Token":"offline-token"}

    def query(self):
        return self.client.get("/settings/deepseek/balance",headers=self.headers)

    def test_official_get_shape_timestamp_and_no_secret(self):
        result=self.query()
        self.assertEqual(result.status_code,200)
        data=result.json()["data"]
        self.assertEqual(data["balance_infos"],payload()["balance_infos"])
        self.assertIn("queried_at",data)
        self.assertEqual(data["http_requests"],1)
        self.assertEqual(data["http_status"],200)
        self.assertNotIn(SECRET,result.text)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(str(self.calls[0].url),"https://api.deepseek.com/user/balance")
        self.assertEqual(self.calls[0].method,"GET")
        self.assertEqual(self.calls[0].headers["Authorization"],"Bearer "+SECRET)

    def test_no_automatic_query_on_health_or_status(self):
        self.client.get("/health")
        self.client.get("/settings/providers",headers=self.headers)
        self.assertEqual(self.calls,[])

    def test_guard_token_and_origin(self):
        self.assertEqual(self.client.get("/settings/deepseek/balance").status_code,403)
        response=self.client.get("/settings/deepseek/balance",headers={**self.headers,"Origin":"https://evil.example"})
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.calls,[])

    def test_missing_and_deleted_key_no_http(self):
        self.keys.environment={}
        self.assertEqual(self.query().status_code,422)
        self.keys.environment={"DEEPSEEK_API_KEY":SECRET}
        self.keys.delete("deepseek")
        self.assertEqual(self.query().status_code,422)
        self.assertEqual(self.calls,[])

    def test_saved_key_priority_and_replacement(self):
        self.storage.write("deepseek","offline-saved-secret",{})
        self.query()
        self.assertEqual(self.calls[-1].headers["Authorization"],"Bearer offline-saved-secret")
        self.storage.write("deepseek","offline-new-secret",{})
        self.query()
        self.assertEqual(self.calls[-1].headers["Authorization"],"Bearer offline-new-secret")

    def test_http_failures_and_redirect_do_not_retry_or_leak(self):
        for status in (401,429,503,302):
            with self.subTest(status=status):
                self.calls.clear()
                self.reply=httpx.Response(status,json={"error":SECRET},headers={"Location":"https://evil.example"})
                response=self.query()
                self.assertEqual(response.status_code,502)
                self.assertNotIn(SECRET,response.text)
                self.assertEqual(len(self.calls),1)

    def test_timeout_one_attempt(self):
        calls=[]
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout(SECRET)
        self.runtime.balance_reader=lambda key:fetch_balance(key,transport=httpx.MockTransport(handler))
        result=self.query()
        self.assertEqual(result.status_code,502)
        self.assertNotIn(SECRET,result.text)
        self.assertEqual(len(calls),1)

    def test_invalid_fields_no_fake_amount(self):
        for field,value in (("total_balance","NaN"),("currency","EUR"),("granted_balance",2.50)):
            with self.subTest(field=field):
                data=payload()
                data["balance_infos"][0][field]=value
                self.reply=httpx.Response(200,json=data)
                result=self.query()
                self.assertEqual(result.status_code,502)
                self.assertNotIn("balance_infos",result.text)

    def test_multiple_currencies_and_unavailable_account(self):
        data=payload()
        data["is_available"]=False
        data["balance_infos"].append({"currency":"USD","total_balance":"0.00","granted_balance":"0","topped_up_balance":"0"})
        self.reply=httpx.Response(200,json=data)
        result=self.query().json()["data"]
        self.assertFalse(result["is_available"])
        self.assertEqual(len(result["balance_infos"]),2)

    def test_busy_does_not_issue_request(self):
        self.runtime.job_lock.acquire()
        try:
            self.assertEqual(self.query().status_code,409)
            self.assertEqual(self.calls,[])
        finally:
            self.runtime.job_lock.release()


if __name__=="__main__":
    unittest.main(verbosity=2)
