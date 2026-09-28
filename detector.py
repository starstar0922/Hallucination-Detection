# -*- coding: utf-8 -*-
"""客服回复幻觉检测器。

- LLMDetector:调用 DeepSeek API(OpenAI 兼容协议,仅用标准库 urllib,零第三方依赖)。
- MockDetector:无 API 时的规则兜底,只覆盖明显的矛盾模式,准确率有限。
统一输出:{"id", "is_hallucination", "type", "severity", "evidence"}
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from taxonomy import TAXONOMY, SEVERITY_ORDER, build_system_prompt

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

ALLOWED_TYPES = set(TAXONOMY)
DEFAULT_SEVERITY = {name: info["severity"] for name, info in TAXONOMY.items()}


def load_api_key() -> str:
    """优先读环境变量 DEEPSEEK_API_KEY,其次读项目根目录 .env 或 config.local.json。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    root = Path(__file__).resolve().parent
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    cfg = root / "config.local.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("api_key", "").strip()
        except (OSError, ValueError):
            return ""
    return ""


class LLMDetector:
    def __init__(self, api_key: str, base_url: str = BASE_URL, model: str = MODEL,
                 max_retries: int = 3, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.system_prompt = build_system_prompt()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _call(self, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 600,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage") or {}
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _call_with_retry(self, user_prompt: str) -> str:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(user_prompt)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                    ValueError, TimeoutError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败(重试 {self.max_retries} 次): {last_err}")

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"回复中未找到 JSON 对象: {text[:200]}")
        return json.loads(text[start:end + 1])

    @staticmethod
    def _normalize(result: dict) -> dict:
        is_h = bool(result.get("is_hallucination", False))
        rtype = str(result.get("type", "")).strip()
        if rtype not in ALLOWED_TYPES:
            rtype = "事实编造" if is_h else "非幻觉"
        severity = str(result.get("severity", "")).strip()
        if severity not in SEVERITY_ORDER:
            severity = DEFAULT_SEVERITY[rtype]
        if not is_h:
            rtype, severity = "非幻觉", "无"
        return {
            "is_hallucination": is_h,
            "type": rtype,
            "severity": severity,
            "evidence": str(result.get("evidence", "")).strip(),
        }

    def detect(self, item: dict) -> dict:
        user_prompt = (
            "请判断以下客服回复是否存在幻觉,并输出 JSON(注意必须输出 JSON 对象):\n"
            + json.dumps({
                "用户问题": item["user_question"],
                "系统回复": item["system_reply"],
                "知识库": item["knowledge_base"],
            }, ensure_ascii=False, indent=2)
        )
        content = self._call_with_retry(user_prompt)
        result = self._normalize(self._parse_json(content))
        result["id"] = item["id"]
        return result


class MockDetector:
    """规则兜底:仅覆盖能力越界、安全误导、优惠编造等明显模式,不覆盖全部类型。"""

    CAPABILITY_CLAIM_PATTERNS = [
        "帮您查", "已帮您", "已修改", "已升级", "已经在处理", "处理中",
        "发到您账户", "专属客服", "预计明天", "已为您",
    ]
    HONEST_PATTERNS = ["无法", "不能", "不支持", "需人工", "转人工"]

    def detect(self, item: dict) -> dict:
        q, r, kb = item["user_question"], item["system_reply"], item["knowledge_base"]

        def verdict(is_h, rtype, evidence):
            severity = DEFAULT_SEVERITY[rtype] if is_h else "无"
            return {"id": item["id"], "is_hallucination": is_h,
                    "type": rtype, "severity": severity, "evidence": evidence}

        # 1) 能力越界:知识库标明无能力,回复却声称已执行
        if ("未接入" in kb or "不具备" in kb or kb.startswith("无")):
            if any(p in r for p in self.CAPABILITY_CLAIM_PATTERNS) and \
               not any(p in r for p in self.HONEST_PATTERNS):
                return verdict(True, "能力越界", f"知识库:{kb[:30]}...;回复声称已执行操作")

        # 2) 安全误导:孕妇问题 + 知识库有警示 + 回复说放心/可用
        if "孕妇" in q and ("孕妇" in kb or "哺乳" in kb):
            if ("放心" in r or "可以" in r) and ("咨询" in kb or "建议" in kb):
                return verdict(True, "安全误导", "知识库提示孕妇需咨询医生,回复却说可用/放心")

        # 3) 政策/优惠编造:知识库明确'无',回复却肯定'有'
        if ("无" in kb or "暂不支持" in kb or "当前无" in kb) and ("优惠" in q or "券" in q or "折扣" in q or "学生" in q):
            if "有的" in r or "可以" in r or "享受" in r:
                return verdict(True, "政策编造", "知识库标明无此优惠,回复却声称有")

        # 4) 参数:知识库'未标注',回复却肯定支持
        if "未标注" in kb and "支持" in r:
            return verdict(True, "事实编造", "知识库未标注该参数,回复却肯定支持")

        return verdict(False, "非幻觉", "")


def create_detector(mock: bool = False):
    if mock:
        return MockDetector()
    key = load_api_key()
    if not key:
        raise SystemExit(
            "未找到 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY,"
            "或创建 config.local.json 写入 {\"api_key\": \"sk-xxx\"};"
            "也可加 --mock 用规则兜底模式运行。"
        )
    return LLMDetector(key)
