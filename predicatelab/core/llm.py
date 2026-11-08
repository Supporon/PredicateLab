"""LLM Client and Structured Proposal Parser for PredicateLab.

Supports DeepSeek API (``https://api.deepseek.com/chat/completions``, models
``deepseek-chat`` and ``deepseek-reasoner``) as well as any OpenAI-compatible
chat completions endpoint using pure Python stdlib ``urllib.request``.

Security & Scientific Protocol Guarantees:
- Never executes arbitrary code from LLM responses: proposals are parsed strictly
  as JSON descriptor objects and must pass domain static validation + probe gates.
- Never persists or prints raw API keys in logs, exceptions, or artifacts.
- Never exposes confirmation or holdout data to the LLM prompt: only discovery
  round feedback (screen/validation fold metrics) is included.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LLMCallResult:
    ok: bool
    model: str
    proposals: list[dict[str, Any]] = field(default_factory=list)
    analysis: str = ""
    reasoning_content: str = ""
    raw_content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""


def _read_dotenv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and v:
                out[k] = v
    except Exception:
        pass
    return out


def resolve_llm_credentials(gen_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve API key, base URL, and model from generator_config, env, or .env files."""
    cfg = dict(gen_cfg or {})
    if cfg.get("enabled") is False:
        return {
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": str(cfg.get("model") or "deepseek-chat"),
            "temperature": float(cfg.get("temperature", 0.7)),
            "max_tokens": int(cfg.get("max_tokens", 8192)),
            "timeout_s": int(cfg.get("timeout_s", 120)),
            "max_retries": int(cfg.get("max_retries", 2)),
        }
    env_map: dict[str, str] = {}
    for candidate_env in (
        REPO_ROOT / ".env",
        Path.cwd() / ".env",
        Path.home() / ".deepseek_env",
    ):
        for k, v in _read_dotenv_file(candidate_env).items():
            env_map.setdefault(k, v)

    key_file = cfg.get("api_key_file") or os.environ.get("DEEPSEEK_API_KEY_FILE")
    file_key = ""
    if key_file and Path(str(key_file)).expanduser().exists():
        try:
            file_key = Path(str(key_file)).expanduser().read_text(encoding="utf-8").strip()
        except Exception:
            file_key = ""
    elif (Path.home() / ".deepseek_api_key").exists():
        try:
            file_key = (Path.home() / ".deepseek_api_key").read_text(encoding="utf-8").strip()
        except Exception:
            file_key = ""

    env_var_name = str(cfg.get("api_key_env") or "DEEPSEEK_API_KEY")
    raw_cfg_key = str(cfg.get("api_key") or "").strip()
    if raw_cfg_key.startswith("${") and raw_cfg_key.endswith("}"):
        env_var_name = raw_cfg_key[2:-1].strip()
        raw_cfg_key = ""

    api_key = (
        raw_cfg_key
        or os.environ.get(env_var_name, "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or file_key
        or env_map.get(env_var_name, "").strip()
        or env_map.get("DEEPSEEK_API_KEY", "").strip()
    )

    base_url = (
        str(cfg.get("base_url") or "").strip()
        or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        or env_map.get("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    )
    model = (
        str(cfg.get("model") or "").strip()
        or os.environ.get("DEEPSEEK_MODEL", "").strip()
        or env_map.get("DEEPSEEK_MODEL", "").strip()
        or "deepseek-chat"
    )
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "temperature": float(cfg.get("temperature", 0.7)),
        "max_tokens": int(cfg.get("max_tokens", 8192)),
        "timeout_s": int(cfg.get("timeout_s", 120)),
        "max_retries": int(cfg.get("max_retries", 2)),
    }


def _repair_json_text(s: str) -> str:
    """Repair common LLM JSON defects: comments, trailing commas, smart quotes."""
    out = s
    out = out.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    out = re.sub(r"//[^\n\r]*", "", out)
    out = re.sub(r"/\*[\s\S]*?\*/", "", out)
    out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def _parse_loose(candidate: str) -> Any:
    """Try strict parse, then a repaired parse."""
    for text in (candidate, _repair_json_text(candidate)):
        try:
            return json.loads(text)
        except Exception:
            continue
    return None


def extract_json_object(text: str) -> dict[str, Any] | list[Any]:
    """Robustly extract JSON object or array from raw LLM output."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty LLM response")
    parsed = _parse_loose(s)
    if parsed is not None:
        return parsed
    # Strip ```json ... ``` blocks
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    for block in fenced:
        parsed = _parse_loose(block.strip())
        if parsed is not None:
            return parsed
    # Outermost balanced { ... } / [ ... ] using strict, then repaired, spans
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first = s.find(open_ch)
        last = s.rfind(close_ch)
        if first != -1 and last > first:
            parsed = _parse_loose(s[first : last + 1])
            if parsed is not None:
                return parsed
    # Last resort: raw_decode from the first brace (handles trailing prose)
    for start_ch in ("{", "["):
        idx = s.find(start_ch)
        if idx == -1:
            continue
        for text in (s[idx:], _repair_json_text(s[idx:])):
            try:
                obj, _end = json.JSONDecoder().raw_decode(text)
                return obj
            except Exception:
                continue
    # Salvage the proposals array of an otherwise-broken object
    m = re.search(r'"proposals"\s*:\s*(\[[\s\S]*)', s)
    if m:
        parsed = _parse_loose(m.group(1))
        if isinstance(parsed, list):
            return {"proposals": parsed}
        # Truncated array: parse every complete top-level object inside it
        objs: list[Any] = []
        depth = 0
        start = -1
        for i, ch in enumerate(m.group(1)):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    parsed_obj = _parse_loose(m.group(1)[start : i + 1])
                    if isinstance(parsed_obj, dict):
                        objs.append(parsed_obj)
                    start = -1
        if objs:
            return {"proposals": objs}
    raise ValueError(f"could not parse JSON from LLM output (len={len(s)})")


class DeepSeekLLMClient:
    """Zero-dependency HTTP client for DeepSeek / OpenAI-compatible Chat Completions."""

    def __init__(self, gen_cfg: dict[str, Any] | None = None):
        creds = resolve_llm_credentials(gen_cfg)
        self.api_key: str = creds["api_key"]
        self.base_url: str = creds["base_url"]
        self.model: str = creds["model"]
        self.temperature: float = creds["temperature"]
        self.max_tokens: int = creds["max_tokens"]
        self.timeout_s: int = creds["timeout_s"]
        self.max_retries: int = creds["max_retries"]

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and len(self.api_key) >= 8)

    def chat_proposals(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        audit_path: Path | None = None,
    ) -> LLMCallResult:
        t0 = time.monotonic()
        if not self.has_api_key:
            res = LLMCallResult(
                ok=False,
                model=self.model,
                latency_s=0.0,
                error="no API key configured (set DEEPSEEK_API_KEY in environment or .env)",
            )
            self._write_audit(audit_path, system_prompt, user_prompt, res)
            return res

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        if "reasoner" not in self.model.lower():
            payload["temperature"] = self.temperature
            payload["response_format"] = {"type": "json_object"}

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "PredicateLab/1.1",
        }

        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_bytes = resp.read()
                resp_json = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                choice = (resp_json.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                raw_content = str(msg.get("content") or "")
                reasoning = str(msg.get("reasoning_content") or "")
                usage = dict(resp_json.get("usage") or {})
                try:
                    parsed = extract_json_object(raw_content)
                except ValueError as exc:
                    self._write_audit(
                        audit_path, system_prompt, user_prompt,
                        LLMCallResult(ok=False, model=self.model, raw_content=raw_content,
                                      usage=usage, latency_s=round(time.monotonic() - t0, 3),
                                      error=f"{exc}; finish_reason={choice.get('finish_reason')}"),
                    )
                    last_err = f"{exc} (finish_reason={choice.get('finish_reason')})"
                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                if isinstance(parsed, dict):
                    proposals = list(parsed.get("proposals") or parsed.get("candidates") or [])
                    analysis = str(parsed.get("round_analysis") or parsed.get("analysis") or "")
                elif isinstance(parsed, list):
                    proposals = list(parsed)
                    analysis = ""
                else:
                    proposals = []
                    analysis = ""
                res = LLMCallResult(
                    ok=True,
                    model=str(resp_json.get("model") or self.model),
                    proposals=[p for p in proposals if isinstance(p, dict)],
                    analysis=analysis,
                    reasoning_content=reasoning,
                    raw_content=raw_content,
                    usage=usage,
                    latency_s=round(time.monotonic() - t0, 3),
                )
                self._write_audit(audit_path, system_prompt, user_prompt, res)
                return res
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_err = f"HTTP {exc.code}: {err_body}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))

        res = LLMCallResult(
            ok=False,
            model=self.model,
            latency_s=round(time.monotonic() - t0, 3),
            error=last_err,
        )
        self._write_audit(audit_path, system_prompt, user_prompt, res)
        return res

    @staticmethod
    def _write_audit(
        audit_path: Path | None,
        system_prompt: str,
        user_prompt: str,
        result: LLMCallResult,
    ) -> None:
        if audit_path is None:
            return
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ok": result.ok,
                "model": result.model,
                "latency_s": result.latency_s,
                "usage": result.usage,
                "error": result.error,
                "analysis": result.analysis,
                "reasoning_content": result.reasoning_content,
                "n_proposals": len(result.proposals),
                "proposals": result.proposals,
                "raw_content": result.raw_content[:60000],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            audit_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
