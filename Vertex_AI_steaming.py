"""
title: Multi-Model Manifold Pipe for Vertex AI (Gemini / Claude / Llama)
author: Feynman Liang + Andres Eufrasio (updated)
version: 0.1.0
license: MIT
requirements: google-auth, requests
"""

import asyncio
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple, Union

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from pydantic import BaseModel, Field

DEBUG = False

# Default sampling params (used only if the request doesn't provide them)
TEMP = 1.0  # Gemini 3 recommends leaving temperature at 1.0
TOPP = 0.95
TOPK = 64
DEFAULT_MAX_TOKENS = 8192  # thinking models need headroom


def _log(*args):
    if DEBUG:
        print(*args)


def _split_models(csv: str) -> List[str]:
    return [m.strip() for m in csv.split(",") if m.strip()]


def _text_of(content) -> str:
    """Flatten OpenAI-style content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _parse_data_url(url: str) -> Optional[Tuple[str, str]]:
    """'data:image/png;base64,XXXX' -> ('image/png', 'XXXX')."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url.split(";base64,", 1)
    return header[len("data:") :], data


def _strip_function_prefix(model_id: str) -> str:
    """Open WebUI prepends "<function_id>." to model IDs. Function IDs contain no
    dots, so drop everything up to the first dot unless the ID already starts
    with a known model family."""
    if "." in model_id and not model_id.startswith(
        ("gemini", "claude", "llama", "meta/")
    ):
        return model_id.split(".", 1)[1]
    return model_id


DEFAULT_PRICING = {
    # USD per 1M tokens. "until" = last day (UTC) a tier applies; no "until" = open-ended.
    "gemini-4-carbon": [
        {"until": "2026-11-30", "in": 2, "out": 10},
        {"in": 4, "out": 20},
    ],
    "gemini-4-argon": [
        {"until": "2026-11-30", "in": 2, "out": 10},
        {"in": 4, "out": 20},
    ],
    "gemini-3.8-flash": [
        {"until": "2026-12-31", "in": 0.75, "out": 3.75},
        {"in": 1.5, "out": 7.5},
    ],
    "gemini-3.7-flash": [
        {"until": "2026-12-31", "in": 0.75, "out": 3.75},
        {"in": 1.5, "out": 7.5},
    ],
    "gemini-3.6-flash": [
        {"until": "2026-12-31", "in": 0.75, "out": 3.75},
        {"in": 1.5, "out": 7.5},
    ],
}


def _usage(prompt=0, completion=0) -> dict:
    prompt, completion = int(prompt or 0), int(completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _usage_sse(usage: dict) -> str:
    """Raw SSE chunk Open WebUI uses to fill the message's info popover."""
    return "data: " + json.dumps({"choices": [], "usage": usage})


def _completion(model: str, text: str, usage: dict) -> dict:
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _gemini_usage(meta: dict) -> dict:
    # thinking tokens are billed as output, so count them as completion tokens
    return _usage(
        meta.get("promptTokenCount"),
        (meta.get("candidatesTokenCount") or 0) + (meta.get("thoughtsTokenCount") or 0),
    )


def _claude_usage(u: dict) -> dict:
    prompt = (
        (u.get("input_tokens") or 0)
        + (u.get("cache_creation_input_tokens") or 0)
        + (u.get("cache_read_input_tokens") or 0)
    )
    return _usage(prompt, u.get("output_tokens"))


class Pipe:
    class Valves(BaseModel):
        PROJECT_ID: str = Field(default="", description="Google Cloud project ID.")
        LOCATION: str = Field(
            default="global",
            description="Region for Gemini. Gemini 3.x previews require 'global'.",
        )
        CLAUDE_LOCATION: str = Field(
            default="us-east5",
            description="Region for Claude models ('global' also works for newer ones).",
        )
        LLAMA_LOCATION: str = Field(
            default="us-central1", description="Region for Llama (MaaS) models."
        )
        SERVICE_ACCOUNT_JSON: str = Field(
            default="", description="Service account JSON (full contents)."
        )
        GEMINI_MODELS: str = Field(
            default="gemini-4-carbon,gemini-4-argon,gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash",
            description="Comma-separated Gemini model IDs.",
        )
        CLAUDE_MODELS: str = Field(
            default="claude-sonnet-4-5@20250929,claude-haiku-4-5@20251001",
            description="Comma-separated Claude model IDs as shown in Vertex Model Garden (blank to disable).",
        )
        LLAMA_MODELS: str = Field(
            default="",
            description="Comma-separated Llama MaaS IDs, e.g. llama-3.3-70b-instruct-maas (blank to disable).",
        )
        GEMINI_SAFETY_THRESHOLD: str = Field(
            default="OFF",
            description="OFF, BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE or BLOCK_LOW_AND_ABOVE.",
        )
        SHOW_USAGE: bool = Field(
            default=True,
            description="Report token usage to Open WebUI (the info popover on each message).",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show live status messages (sending / generating / done summary).",
        )
        SHOW_COST: bool = Field(
            default=True,
            description="Add an estimated USD cost to the usage info and the done status.",
        )
        USD_TO_GBP: float = Field(
            default=0.75,
            description="USD to GBP rate used to convert estimated costs (prices are in USD).",
        )
        PRICING_JSON: str = Field(
            default=json.dumps(DEFAULT_PRICING),
            description=(
                "USD per 1M tokens by model-ID prefix: "
                '{"model-prefix": [{"until": "YYYY-MM-DD", "in": 1, "out": 5}, {"in": 2, "out": 10}]}. '
                "First tier whose 'until' hasn't passed is used. Models not listed show no cost."
            ),
        )

    def __init__(self):
        self.id = "vertex_ai_streaming"  # must match this function's ID in OpenWebUI
        self.type = "manifold"
        self.name = "Vertex "
        self.valves = self.Valves()
        self.credentials = None
        self.authed_session: Optional[AuthorizedSession] = None
        self._cred_hash = None

    # ------------------------------------------------------------------ auth
    def _ensure_auth(self):
        sa_json = self.valves.SERVICE_ACCOUNT_JSON
        if not sa_json:
            raise ValueError("SERVICE_ACCOUNT_JSON is not configured in Valves.")
        h = hashlib.sha256(sa_json.encode()).hexdigest()
        if self.authed_session and self._cred_hash == h:
            return  # AuthorizedSession refreshes expired tokens itself
        try:
            info = json.loads(sa_json)
            self.credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.authed_session = AuthorizedSession(self.credentials)
            self._cred_hash = h
            _log("Authentication successful.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding SERVICE_ACCOUNT_JSON: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to authenticate with Google Cloud: {e}")

    # ----------------------------------------------------------------- models
    def pipes(self) -> List[dict]:
        models = []
        for m in _split_models(self.valves.GEMINI_MODELS):
            models.append({"id": m, "name": m})
        for m in _split_models(self.valves.CLAUDE_MODELS):
            models.append({"id": m, "name": m.split("@")[0]})
        for m in _split_models(self.valves.LLAMA_MODELS):
            models.append({"id": m, "name": m})
        return models

    # ------------------------------------------------------------------ utils
    def _base(self, location: str, api: str = "v1") -> str:
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/{api}/projects/{self.valves.PROJECT_ID}"
            f"/locations/{location}"
        )

    @staticmethod
    def _sse_json(response) -> Iterator[dict]:
        """Yield parsed JSON objects from an SSE response."""
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError as e:
                _log(f"Bad SSE chunk: {e}: {payload[:200]}")

    # ------------------------------------------------------- cost / status utils
    def _estimate_cost(self, model_id: str, usage: dict) -> Optional[float]:
        try:
            table = json.loads(self.valves.PRICING_JSON or "{}")
        except json.JSONDecodeError:
            return None
        keys = [k for k in table if model_id.startswith(k)]
        if not keys:
            return None
        today = datetime.now(timezone.utc).date().isoformat()
        for tier in table[max(keys, key=len)]:
            if "until" not in tier or today <= tier["until"]:
                return (
                    (usage.get("prompt_tokens") or 0) * tier["in"]
                    + (usage.get("completion_tokens") or 0) * tier["out"]
                ) / 1e6
        return None

    def _add_cost(self, model_id: str, usage: dict) -> dict:
        if self.valves.SHOW_COST:
            cost = self._estimate_cost(model_id, usage)
            if cost is not None:
                usage = {
                    **usage,
                    "estimated_cost_gbp": round(cost * self.valves.USD_TO_GBP, 6),
                    "estimated_cost_usd": round(cost, 6),
                }
        return usage

    async def _emit(self, emitter, text: str, done: bool = False):
        if not emitter or not self.valves.SHOW_STATUS:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": text, "done": done}}
            )
        except Exception as e:  # never let a status failure break the reply
            _log("status emit failed:", e)

    @staticmethod
    def _summary(usage: Optional[dict], started: float) -> str:
        parts = ["Done"]
        if usage:
            parts.append(
                f"{usage.get('prompt_tokens', 0):,} in / "
                f"{usage.get('completion_tokens', 0):,} out"
            )
            if usage.get("estimated_cost_gbp") is not None:
                parts.append(f"~£{usage['estimated_cost_gbp']:.4f}")
        parts.append(f"{time.monotonic() - started:.1f}s")
        return " · ".join(parts)

    async def _astream(self, it, model_id: str, emitter, started: float):
        """Wrap the blocking sync generator so the event loop (and status
        updates) stay responsive, and attach cost to the usage chunk."""
        sentinel = object()
        usage = None
        first = True
        try:
            while True:
                item = await asyncio.to_thread(next, it, sentinel)
                if item is sentinel:
                    break
                if first:
                    first = False
                    await self._emit(emitter, f"Generating with {model_id}…")
                if isinstance(item, str) and item.startswith("data: "):
                    try:
                        d = json.loads(item[len("data: ") :])
                        if d.get("usage"):
                            usage = self._add_cost(model_id, d["usage"])
                            d["usage"] = usage
                            item = "data: " + json.dumps(d)
                    except json.JSONDecodeError:
                        pass
                yield item
        finally:
            # always clear the "Sending/Generating" status, even if the stream
            # is closed early by the client
            await self._emit(emitter, self._summary(usage, started), done=True)

    # ------------------------------------------------------------------- main
    async def pipe(self, body: dict, __event_emitter__=None, __task__=None):
        started = time.monotonic()
        model_id = _strip_function_prefix(body.get("model") or "")
        # Background jobs (title / tags / follow-up generation) reuse the chat's
        # emitter and would overwrite the real reply's status, so stay quiet.
        emitter = None if __task__ else __event_emitter__
        await self._emit(emitter, f"Sending to {model_id}…")

        result = await asyncio.to_thread(self._pipe_sync, body)

        if isinstance(result, str):
            ok = not result.startswith("Error")
            await self._emit(
                emitter,
                self._summary(None, started) if ok else result[:120],
                done=True,
            )
            return result
        if isinstance(result, dict):
            usage = result.get("usage")
            if usage:
                usage = self._add_cost(model_id, usage)
                result["usage"] = usage
            await self._emit(emitter, self._summary(usage, started), done=True)
            return result
        return self._astream(result, model_id, emitter, started)

    def _pipe_sync(self, body: dict) -> Union[str, dict, Iterator[str]]:
        try:
            self._ensure_auth()
        except (ValueError, ConnectionError) as e:
            return f"Error: {e}"

        if not self.valves.PROJECT_ID:
            return "Error: PROJECT_ID is not configured in Valves."

        model_id = body.get("model")
        if not model_id:
            return "Error: No model specified."

        model_id = _strip_function_prefix(model_id)

        try:
            if model_id.startswith("gemini"):
                return self.call_gemini_model(body, model_id)
            if model_id.startswith("claude"):
                return self.call_claude_model(body, model_id)
            if model_id.startswith("llama") or model_id.startswith("meta/"):
                return self.call_llama_model(body, model_id)
            return f"Error: Unknown or unsupported model ID: {model_id}"
        except Exception as e:
            if DEBUG:
                import traceback

                traceback.print_exc()
            return f"Error: Unexpected failure for model {model_id}: {e}"

    # ----------------------------------------------------------------- Gemini
    def _gemini_parts(self, content) -> List[dict]:
        if isinstance(content, str):
            return [{"text": content}] if content else []
        parts = []
        for p in content or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and p.get("text"):
                parts.append({"text": p["text"]})
            elif p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                parsed = _parse_data_url(url)
                if parsed:
                    parts.append(
                        {"inlineData": {"mimeType": parsed[0], "data": parsed[1]}}
                    )
        return parts

    def format_gemini_contents(self, messages) -> Tuple[List[dict], Optional[dict]]:
        """Return (contents, systemInstruction). Merges consecutive same-role turns."""
        system_texts: List[str] = []
        contents: List[dict] = []
        for msg in messages:
            role, content = msg.get("role"), msg.get("content")
            if role == "system":
                t = _text_of(content)
                if t:
                    system_texts.append(t)
                continue
            parts = self._gemini_parts(content)
            if not parts:
                continue
            api_role = "model" if role == "assistant" else "user"
            if contents and contents[-1]["role"] == api_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": api_role, "parts": parts})
        system = (
            {"parts": [{"text": "\n\n".join(system_texts)}]} if system_texts else None
        )
        return contents, system

    def call_gemini_model(self, body, model_id):
        location = self.valves.LOCATION or "global"
        stream = body.get("stream", False)
        # v1beta1 serves both GA and preview models
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = (
            f"{self._base(location, 'v1beta1')}/publishers/google/models/"
            f"{model_id}:{method}"
        )

        contents, system = self.format_gemini_contents(body["messages"])
        if not contents:
            return "Error: No non-empty messages to send."

        threshold = self.valves.GEMINI_SAFETY_THRESHOLD
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": body.get("temperature", TEMP),
                "maxOutputTokens": body.get("max_tokens", DEFAULT_MAX_TOKENS),
                "topP": body.get("top_p", TOPP),
                "topK": body.get("top_k", TOPK),
            },
            "safetySettings": [
                {"category": c, "threshold": threshold}
                for c in (
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_HARASSMENT",
                )
            ],
        }
        if system:
            data["systemInstruction"] = system
        if body.get("stop"):
            stop = body["stop"]
            data["generationConfig"]["stopSequences"] = (
                [stop] if isinstance(stop, str) else stop
            )

        _log("Gemini URL:", url)

        if stream:
            return self._stream_gemini_response(url, data)

        resp = self.authed_session.post(url, json=data)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        rdata = resp.json()
        text = self._parse_gemini_chunk(rdata, final=True) or ""
        if (
            self.valves.SHOW_USAGE
            and rdata.get("usageMetadata")
            and not text.startswith("Error:")
        ):
            return _completion(model_id, text, _gemini_usage(rdata["usageMetadata"]))
        return text

    def _parse_gemini_chunk(self, chunk: dict, final: bool = False) -> Optional[str]:
        """Extract visible text from one Gemini response object (or an error string)."""
        reason = chunk.get("promptFeedback", {}).get("blockReason")
        if reason:
            return f"Error: Request blocked by prompt safety filters: {reason}"

        cands = chunk.get("candidates") or []
        if not cands:
            return "Error: No candidates returned by the model." if final else None
        cand = cands[0]

        parts = (cand.get("content") or {}).get("parts") or []
        # Skip "thought" parts (Gemini 3 / 2.5 thinking summaries)
        text = "".join(
            p.get("text", "") for p in parts if "text" in p and not p.get("thought")
        )

        finish = cand.get("finishReason")
        if finish in (
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "RECITATION",
        ):
            note = f"\n\nError: generation stopped ({finish})."
            return (text + note) if text else note.strip()
        if final and not text:
            return f"Error: Empty response (finishReason={finish})."
        return text

    def _stream_gemini_response(self, url: str, data: dict) -> Iterator[str]:
        resp = self.authed_session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )
        if resp.status_code != 200:
            yield f"Error: {resp.status_code} {resp.text}"
            return
        meta: dict = {}
        for chunk in self._sse_json(resp):
            meta = chunk.get("usageMetadata") or meta
            text = self._parse_gemini_chunk(chunk)
            if text:
                yield text
        if self.valves.SHOW_USAGE and meta:
            yield _usage_sse(_gemini_usage(meta))

    # ----------------------------------------------------------------- Claude
    def _claude_content(self, content):
        if isinstance(content, str):
            return content
        blocks = []
        for p in content or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and p.get("text"):
                blocks.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image_url":
                parsed = _parse_data_url((p.get("image_url") or {}).get("url", ""))
                if parsed:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": parsed[0],
                                "data": parsed[1],
                            },
                        }
                    )
        return blocks

    def format_claude_messages(self, messages) -> Tuple[List[dict], Optional[str]]:
        system_parts: List[str] = []
        out: List[dict] = []
        for msg in messages:
            role, content = msg.get("role"), msg.get("content")
            if role == "system":
                t = _text_of(content)
                if t:
                    system_parts.append(t)
                continue
            if role not in ("user", "assistant"):
                continue
            c = self._claude_content(content)
            if not c:
                continue
            if out and out[-1]["role"] == role:
                # merge consecutive same-role turns
                prev = out[-1]["content"]
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                if isinstance(c, str):
                    c = [{"type": "text", "text": c}]
                out[-1]["content"] = prev + c
            else:
                out.append({"role": role, "content": c})
        # Claude requires the first message to be from the user
        if out and out[0]["role"] == "assistant":
            out.insert(0, {"role": "user", "content": "(conversation start)"})
        return out, ("\n\n".join(system_parts) if system_parts else None)

    def call_claude_model(self, body, model_id):
        location = self.valves.CLAUDE_LOCATION or "us-east5"
        stream = body.get("stream", False)
        method = "streamRawPredict" if stream else "rawPredict"
        url = f"{self._base(location)}/publishers/anthropic/models/{model_id}:{method}"

        messages, system = self.format_claude_messages(body["messages"])
        if not messages:
            return "Error: No non-empty messages to send."

        # rawPredict takes the native Anthropic Messages body (model is in the URL)
        payload = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": messages,
            "max_tokens": body.get("max_tokens", DEFAULT_MAX_TOKENS),
            "stream": stream,
        }
        if system:
            payload["system"] = system
        # Newer Claude models reject temperature + top_p together, so send only temperature
        if "temperature" in body:
            payload["temperature"] = body["temperature"]
        if body.get("stop"):
            stop = body["stop"]
            payload["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        _log("Claude URL:", url)

        if stream:
            return self._stream_claude_response(url, payload)

        resp = self.authed_session.post(url, json=payload)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        data = resp.json()
        text = "".join(
            b.get("text", "")
            for b in data.get("content", [])
            if b.get("type") == "text"
        )
        if data.get("stop_reason") == "max_tokens":
            text += "\n\n[Output truncated: max_tokens reached]"
        if not text:
            return "Error: No text content found in the model response."
        if self.valves.SHOW_USAGE and data.get("usage"):
            return _completion(model_id, text, _claude_usage(data["usage"]))
        return text

    def _stream_claude_response(self, url: str, payload: dict) -> Iterator[str]:
        resp = self.authed_session.post(
            url, json=payload, stream=True, headers={"Accept": "text/event-stream"}
        )
        if resp.status_code != 200:
            yield f"Error: {resp.status_code} {resp.text}"
            return
        usage: dict = {}
        for ev in self._sse_json(resp):
            t = ev.get("type")
            if t == "message_start":
                usage.update((ev.get("message") or {}).get("usage") or {})
            elif t == "content_block_delta":
                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")
            elif t == "message_delta":
                usage.update(ev.get("usage") or {})
                if ev.get("delta", {}).get("stop_reason") == "max_tokens":
                    yield "\n\n[Output truncated: max_tokens reached]"
            elif t == "message_stop":
                break
            elif t == "error":
                err = ev.get("error", {})
                yield f"Error from Claude stream: {err.get('type')} - {err.get('message')}"
                return
        if self.valves.SHOW_USAGE and usage:
            yield _usage_sse(_claude_usage(usage))

    # ------------------------------------------------------------------ Llama
    def call_llama_model(self, body, model_id):
        """Llama / other MaaS models via Vertex's OpenAI-compatible endpoint."""
        location = self.valves.LLAMA_LOCATION or "us-central1"
        url = f"{self._base(location)}/endpoints/openapi/chat/completions"
        stream = body.get("stream", False)

        full_id = model_id if "/" in model_id else f"meta/{model_id}"
        messages = []
        for m in body["messages"]:
            t = _text_of(m.get("content"))
            if t and m.get("role") in ("system", "user", "assistant"):
                messages.append({"role": m["role"], "content": t})

        data = {
            "model": full_id,
            "messages": messages,
            "temperature": body.get("temperature", 0.6),
            "max_tokens": body.get("max_tokens", 2048),
            "top_p": body.get("top_p", 0.9),
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }
        _log("Llama URL:", url)

        if stream:
            return self._stream_openai_response(url, data)

        resp = self.authed_session.post(url, json=data)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        rdata = resp.json()
        choices = rdata.get("choices") or []
        if not choices:
            return "Error: No response generated (no choices)."
        text = (choices[0].get("message") or {}).get("content", "") or ""
        if choices[0].get("finish_reason") == "length":
            text += "\n\n[Output truncated: max_tokens reached]"
        u = rdata.get("usage") or {}
        if self.valves.SHOW_USAGE and u:
            return _completion(
                model_id,
                text,
                _usage(u.get("prompt_tokens"), u.get("completion_tokens")),
            )
        return text

    def _stream_openai_response(self, url: str, data: dict) -> Iterator[str]:
        resp = self.authed_session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )
        if resp.status_code != 200:
            yield f"Error: {resp.status_code} {resp.text}"
            return
        usage: dict = {}
        for chunk in self._sse_json(resp):
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content
        if self.valves.SHOW_USAGE and usage:
            yield _usage_sse(
                _usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
            )
