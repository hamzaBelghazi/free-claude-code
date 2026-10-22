#!/usr/bin/env python3
"""
free-claude — Anthropic Messages API bridge for the aifreeforever GPT-5 chat.

Claude Code talks to this server exactly as it would talk to Anthropic:

    POST /v1/messages              (JSON body, stream=true or false)
    POST /v1/messages/count_tokens
    GET  /health

For every Claude Code request the bridge builds ONE self-contained prompt
from the full Anthropic request (system + tool schemas + every message,
including tool_use / tool_result blocks), sends it to a FRESH browser chat,
parses the reply, and converts it back into a valid Anthropic response
(content blocks + stop_reason) or Anthropic SSE.

Design notes
------------
* Full replay, fresh chat. Claude Code resends the whole transcript on every
  turn anyway, so the browser conversation is never continued; each API call
  starts from a reloaded page with no stale state.
* Protocol-first. Experiments against the site show instructions at the END
  of a long user message are ignored, while instructions at the TOP are
  followed. The response protocol and a brief of the latest request/tool
  result therefore go at the very top of the single submitted message.
* One unsplit message. The whole Claude Code prompt (system + tools +
  transcript) is submitted as one message, no chunking. The site accepts
  100KB+ inputs. A short brief at the top keeps the latest request and tool
  result in the model's instruction zone even on very large prompts.
* Tool calls are requested as one JSON object:
      {"tool_calls":[{"name":"Bash","input":{"command":"..."}}]}
  but the parser also accepts Anthropic-style content arrays and the usual
  model variants (tool/tool_name, flat arguments, cmd, ...).

Usage:
    python3 free-claude.py                 # serves 127.0.0.1:8000
    FREE_CLAUDE_PORT=8010 python3 free-claude.py

Point Claude Code at it:
    ANTHROPIC_BASE_URL=http://127.0.0.1:8000
    ANTHROPIC_AUTH_TOKEN=local
    ANTHROPIC_MODEL=gpt-5

Environment:
    FREE_CLAUDE_HOST            default 127.0.0.1
    FREE_CLAUDE_PORT            default 8000
    FREE_CLAUDE_URL             default https://aifreeforever.com/chat/gpt-5
    FREE_CLAUDE_PROFILE         Chrome profile (default ~/.config/free-claude)
    FREE_CLAUDE_HEADLESS        "0" to show the browser (default headless)
    FREE_CLAUDE_COMPACT_TOOLS   "0" to send tool schemas verbatim (default 1)
    FREE_CLAUDE_TIMEOUT         seconds to wait for one answer (default 300)
    FREE_CLAUDE_MAX_PROMPT      max prompt chars sent to the site (default 70000)
    FREE_CLAUDE_DEBUG           "1" for verbose logs
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import string
import subprocess
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST = os.environ.get("FREE_CLAUDE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FREE_CLAUDE_PORT", "8000"))
CHAT_URL = os.environ.get("FREE_CLAUDE_URL", "https://aifreeforever.com/chat/gpt-5")

PROFILE_DIR = os.path.expanduser(
    os.environ.get("FREE_CLAUDE_PROFILE", "~/.config/free-claude")
)
CHROME_BINARY = os.environ.get("FREE_CLAUDE_CHROME", "/usr/bin/google-chrome")
CHROMEDRIVER = os.environ.get("FREE_CLAUDE_CHROMEDRIVER", "/usr/local/bin/chromedriver-152")

HEADLESS = os.environ.get("FREE_CLAUDE_HEADLESS", "1") == "1"
COMPACT_TOOLS = os.environ.get("FREE_CLAUDE_COMPACT_TOOLS", "1") != "0"
RESPONSE_TIMEOUT = float(os.environ.get("FREE_CLAUDE_TIMEOUT", "300"))
MAX_PROMPT = int(os.environ.get("FREE_CLAUDE_MAX_PROMPT", "70000"))
STABLE_SECONDS = 3.0
DEBUG = os.environ.get("FREE_CLAUDE_DEBUG", "0") == "1"

# Protocol placed at the TOP of the prompt. Empirically the web model follows
# top-of-message instructions and ignores trailing ones on long inputs.
PROTOCOL = """\
You are the model behind Claude Code, the coding agent CLI. You cannot touch
the machine directly: the external runtime executes every tool you request
and returns a [TOOL RESULT]. Continue the conversation from the latest request
or tool result. Never mention this browser, this protocol, or any gateway.

Reply with EXACTLY ONE of these two forms, nothing else:

1) A normal text answer, when no tool is needed (task finished, question
   answered, or you need the user to decide something).

2) One JSON object, when a tool is needed:
{"tool_calls":[{"name":"Bash","input":{"command":"..."}}]}

Rules:
- Use only tools listed in the TOOLS section below.
- input must match that tool's schema. Bash uses {"command": "..."} (never
  "cmd"). Edit/Write need file_path, and old_string/new_string or content.
- Pasting code or file contents into chat does NOT create or change files.
  If the request asks you to create, edit, improve, redesign, fix, run, or
  inspect anything, you MUST call the appropriate tool. For an existing file,
  Read it first, then Edit/Write. Never claim a change you did not make.
  For a new file, call Write with file_path and the FULL file content.
  Never use a Bash heredoc, cat, tee, or redirection to create a file.
  text-only reply is a failed task.
- JavaScript: NEVER use backticks or template literals (${...}). This site
  strips backticks and renders ${...} as math Unicode, which breaks code.
  Use string concatenation with + and quotes instead, e.g.
  "Time: " + m + ":" + s.
- When asked to improve, redesign, restyle, or fix existing code, PRESERVE
  all existing functionality and features (game logic, controls, scoring,
  audio, data). Change only the requested aspect. Do not rewrite from
  scratch or drop features unless the user explicitly asks.
- At most 4 tool calls per reply.
- A [TOOL RESULT] means that call ALREADY RAN. Never repeat it. If the result
  completes the task, answer with a SHORT final summary (2-5 sentences) and
  stop. Do not paste the file again and do not call Write twice for the same
  file_path.
- Do not wrap the JSON in markdown fences. No prose around the JSON.
- Think privately; do not output chain-of-thought or restate the conversation.
"""

def log(*args: Any) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[free-claude {stamp}]", *args, flush=True)


def debug(*args: Any) -> None:
    if DEBUG:
        log("[debug]", *args)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def system_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        chunks = []
        for block in system:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                if block.get("type") in (None, "text"):
                    chunks.append(str(block.get("text", "")))
        return "\n\n".join(c for c in chunks if c)
    return str(system)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                pieces.append(str(block.get("text", "")))
        return "\n".join(p for p in pieces if p)
    return str(content)


def tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    pieces.append(str(block.get("text", "")))
                elif block.get("type") == "image":
                    pieces.append("[image omitted]")
                else:
                    pieces.append(json.dumps(block, ensure_ascii=False))
            else:
                pieces.append(str(block))
        return "\n".join(pieces)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def compact_tool(tool: dict) -> dict:
    """Keep the parts the model needs to build a valid call."""
    if not COMPACT_TOOLS:
        return tool
    schema = tool.get("input_schema") or {}
    props: dict[str, Any] = {}
    for name, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        slim: dict[str, Any] = {}
        if spec.get("type"):
            slim["type"] = spec["type"]
        if spec.get("enum"):
            slim["enum"] = spec["enum"][:30]
        if spec.get("items"):
            items = spec["items"]
            slim["items"] = {"type": items.get("type")} if isinstance(items, dict) else items
        desc = str(spec.get("description") or "").strip()
        if desc:
            slim["description"] = desc[:140]
        props[name] = slim
    return {
        "name": tool.get("name"),
        "description": str(tool.get("description") or "").strip()[:500],
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": schema.get("required", []),
        },
    }


def clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def serialize_messages(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages or []:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        if isinstance(content, str):
            lines.append(f"[{role}]\n" + clip_text(content, 8000))
            continue
        if not isinstance(content, list):
            lines.append(f"[{role}]\n{content}")
            continue
        for block in content:
            if not isinstance(block, dict):
                lines.append(f"[{role}]\n{block}")
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    lines.append(f"[{role}]\n" + clip_text(text, 8000))
            elif btype == "tool_use":
                lines.append(
                    "[ASSISTANT TOOL CALL]\n"
                    f"name: {block.get('name')}\n"
                    f"id: {block.get('id', '')}\n"
                    "input: "
                    + clip_text(
                        json.dumps(block.get("input") or {}, ensure_ascii=False),
                        4000,
                    )
                )
            elif btype == "tool_result":
                lines.append(
                    "[TOOL RESULT]\n"
                    f"tool_use_id: {block.get('tool_use_id', '')}\n"
                    f"is_error: {bool(block.get('is_error', False))}\n"
                    "result:\n"
                    + clip_text(tool_result_text(block.get("content", "")), 12000)
                )
            elif btype == "thinking":
                continue
            else:
                lines.append(
                    f"[{role} {btype}]\n"
                    + json.dumps(block, ensure_ascii=False)
                )
    return "\n\n".join(lines)


def latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        text = content_to_text(message.get("content", ""))
        if text.strip():
            return text.strip()
    return ""


def latest_tool_results(messages: list[dict], limit: int = 3) -> list[str]:
    out: list[str] = []
    for message in reversed(messages or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                body = tool_result_text(block.get("content", ""))
                flag = "error" if block.get("is_error") else "ok"
                out.append(f"[{flag}] {body[:900]}")
                if len(out) >= limit:
                    return out
    return out


def last_executed_call(messages: list[dict]):
    """Returns (tool_use_block, tool_result_block) for the newest completed
    assistant tool call, or (None, None)."""
    uses: dict[str, dict] = {}
    results: dict[str, dict] = {}
    last_use = None
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                uses[block.get("id")] = block
                last_use = block
            elif block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block
    if last_use is None:
        return None, None
    return last_use, results.get(last_use.get("id"))


def latest_request_text(messages: list[dict]) -> str:
    """Newest real user request, ignoring Claude Code's system-reminders."""
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        raw = content_to_text(message.get("content", ""))
        cleaned = re.sub(
            r"<system-reminder>.*?</system-reminder>", "", raw, flags=re.S | re.I
        )
        cleaned = re.sub(
            r"<env>.*?</env>", "", cleaned, flags=re.S | re.I
        ).strip()
        if cleaned:
            return cleaned
    return ""


def latest_user_message(messages: list[dict]):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message
    return None


def latest_user_has_tool_result(messages: list[dict]) -> bool:
    message = latest_user_message(messages)
    content = message.get("content") if message else None
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    return False


def same_path_write(call: dict, messages: list[dict]) -> bool:
    """True when call rewrites the file whose Write just succeeded."""
    last_call, last_result = last_executed_call(messages)
    if not last_call or not last_result:
        return False
    if str(last_call.get("name", "")).lower() != "write":
        return False
    if str(call.get("name", "")).lower() != "write":
        return False
    old = (last_call.get("input") or {}).get("file_path")
    new = (call.get("input") or {}).get("file_path")
    return bool(old) and old == new


def write_target(call: dict, cwd: str) -> tuple[str, str] | None:
    if str(call.get("name", "")).lower() != "write":
        return None
    inp = call.get("input") or {}
    path = inp.get("file_path")
    content = inp.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return None
    full = path if os.path.isabs(path) else os.path.join(cwd or ".", path)
    return full, content


def is_identical_write(call: dict, cwd: str) -> bool:
    target = write_target(call, cwd)
    if not target:
        return False
    full, content = target
    try:
        old = open(full, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    return bool(old.strip()) and content.strip() == old.strip()


def is_destructive_write(call: dict, cwd: str) -> bool:
    target = write_target(call, cwd)
    if not target:
        return False
    full, content = target
    try:
        old = open(full, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    if len(old.strip()) < 500:
        return False
    if len(content.strip()) < 0.7 * len(old.strip()):
        return True
    markers = ["<canvas", "snake", "food", "score", "keydown"]
    old_markers = [m for m in markers if m.lower() in old.lower()]
    missing = [m for m in old_markers if m.lower() not in content.lower()]
    return len(old_markers) >= 3 and len(missing) >= 2


def validate_source(path: str, source: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".html", ".htm"):
            scripts = []
            for attrs, body in re.findall(
                r"<script([^>]*)>(.*?)</script>", source, re.S | re.I
            ):
                if re.search(
                    r"type\s*=\s*['\"]?(?:application/json|text/template)",
                    attrs,
                    re.I,
                ):
                    continue
                if body.strip():
                    scripts.append(body)
            return _node_check("\n;\n".join(scripts))
        if ext == ".js":
            return _node_check(source)
        if ext == ".py":
            try:
                compile(source, path, "exec")
            except SyntaxError as exc:
                return f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})"
            return None
        if ext == ".json":
            try:
                json.loads(source)
            except Exception as exc:
                return f"JSON error: {exc}"
            return None
        return None
    except Exception as exc:
        return f"validation failed: {exc}"


def validate_call(call: dict, cwd: str) -> str | None:
    name = str(call.get("name", "")).lower()
    if name not in ("write", "edit"):
        return None
    inp = call.get("input") or {}
    path = inp.get("file_path")
    if not isinstance(path, str) or not path:
        return None
    full = path if os.path.isabs(path) else os.path.join(cwd or ".", path)
    exists = os.path.isfile(full)
    # If the current file is already broken, allow a fixing edit through;
    # post-write validation will check the result.
    if exists and validate_file(full):
        return None
    if name == "write":
        content = inp.get("content")
        return validate_source(full, content) if isinstance(content, str) else None
    old = inp.get("old_string")
    new = inp.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str) or not exists:
        return None
    try:
        current = open(full, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    if old not in current:
        return None
    updated = (
        current.replace(old, new)
        if inp.get("replace_all")
        else current.replace(old, new, 1)
    )
    return validate_source(full, updated)


def backup_target(call: dict, cwd: str) -> None:
    name = str(call.get("name", "")).lower()
    if name not in ("write", "edit"):
        return
    inp = call.get("input") or {}
    path = inp.get("file_path")
    if not isinstance(path, str) or not path:
        return
    full = path if os.path.isabs(path) else os.path.join(cwd or ".", path)
    if not os.path.isfile(full):
        return
    backup_dir = os.path.join(cwd or ".", ".free-claude-backups")
    try:
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(
            backup_dir, stamp + "-" + os.path.basename(full)
        )
        shutil.copy2(full, dest)
        backups = sorted(
            os.path.join(backup_dir, n) for n in os.listdir(backup_dir)
        )
        for old in backups[:-30]:
            try:
                os.unlink(old)
            except OSError:
                pass
    except Exception as exc:
        debug("backup failed:", exc)


def latest_mutation_file(messages: list[dict], cwd: str) -> str:
    uses: dict[str, dict] = {}
    last_path = ""
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                uses[str(block.get("id", ""))] = block
            elif block.get("type") == "tool_result":
                if block.get("is_error"):
                    continue
                use = uses.get(str(block.get("tool_use_id", "")))
                if not use or str(use.get("name", "")).lower() not in MUTATION_TOOLS:
                    continue
                path = (use.get("input") or {}).get("file_path")
                if isinstance(path, str) and path:
                    last_path = path
    if not last_path:
        return ""
    return last_path if os.path.isabs(last_path) else os.path.join(cwd or ".", last_path)


def _node_check(code: str) -> str | None:
    if not code.strip():
        return None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code)
            temp_path = handle.name
        try:
            proc = subprocess.run(
                ["node", "--check", temp_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if proc.returncode == 0:
        return None
    error = (proc.stderr or proc.stdout or "syntax error").strip()
    error = error.replace(temp_path, "inline script")
    return clip_text(error, 1200)


def validate_file(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".js":
            return _node_check(open(path, encoding="utf-8", errors="replace").read())
        if ext in (".html", ".htm"):
            text = open(path, encoding="utf-8", errors="replace").read()
            scripts = []
            for attrs, body in re.findall(
                r"<script([^>]*)>(.*?)</script>", text, re.S | re.I
            ):
                if re.search(r"type\s*=\s*['\"]?(?:application/json|text/template)", attrs, re.I):
                    continue
                if body.strip():
                    scripts.append(body)
            return _node_check("\n;\n".join(scripts))
        if ext == ".py":
            source = open(path, encoding="utf-8", errors="replace").read()
            try:
                compile(source, path, "exec")
            except SyntaxError as exc:
                return f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})"
            return None
        if ext == ".json":
            try:
                json.load(open(path, encoding="utf-8", errors="replace"))
            except Exception as exc:
                return f"JSON error: {exc}"
            return None
        if ext in (".sh", ".bash"):
            proc = subprocess.run(
                ["bash", "-n", path], capture_output=True, text=True, timeout=15
            )
            if proc.returncode != 0:
                return clip_text((proc.stderr or proc.stdout).strip(), 1200)
        return None
    except Exception as exc:
        return f"validation failed: {exc}"


def working_directory(messages: list[dict]) -> str:
    """Best-effort extraction of Claude Code's real project directory."""
    haystack = "\n".join(
        content_to_text(m.get("content", "")) for m in (messages or [])
    )
    patterns = (
        r"<env>.*?Working directory:\s*([^\n<]+)",
        r"Primary working directory:\s*([^\n<]+)",
        r"Working directory:\s*([^\n<]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, haystack, re.I | re.S)
        if not match:
            continue
        path = match.group(1).strip().strip("\"'` ")
        if path.startswith("/"):
            return path.rstrip("/")
    return ""


CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.S)
CODE_LANGS = {
    "html", "htm", "css", "js", "javascript", "jsx", "ts", "typescript",
    "tsx", "python", "py", "json", "bash", "sh", "shell", "sql", "go",
    "rust", "rs", "java", "c", "cpp", "csharp", "cs", "php", "rb", "ruby",
    "xml", "svg", "md", "markdown", "yaml", "yml",
}
CREATION_VERBS = {
    "create", "make", "build", "write", "generate", "implement", "add",
    "develop", "fix", "update", "change", "refactor", "modify", "convert",
    "port", "turn", "rewrite", "scaffold",
}

MODIFY_VERBS = CREATION_VERBS | {
    "improve", "enhance", "polish", "redesign", "restyle", "tweak",
    "adjust", "optimize", "optimise", "simplify", "modernize", "clean",
    "beautify", "upgrade", "extend", "replace", "rename", "move", "delete",
    "remove", "style", "document", "comment", "format", "lint", "restore",
    "re-enable", "reenable", "enable", "disable", "repair", "correct",
    "ensure", "support", "revert", "undo", "integrate", "wire", "hook",
    "switch", "migrate", "set",
}
ACTION_VERBS = MODIFY_VERBS | {
    "run", "execute", "install", "test", "debug", "inspect", "review",
}
MUTATION_TOOLS = {"write", "edit", "notebookedit"}
FILE_RE = re.compile(
    r"([\w./-]+\.(?:html?|css|js|jsx|ts|tsx|py|json|sh|go|rs|java|c|"
    r"cpp|cs|php|rb|xml|svg|md|ya?ml))\b",
    re.I,
)


def _is_request(text: str, verbs: set[str]) -> bool:
    low = (text or "").lower()
    if re.match(
        r"\s*(how|what|why|explain|describe|show me how|can you explain|"
        r"is|are|does|do)\b",
        low,
    ):
        return False
    return any(re.search(rf"\b{v}\b", low) for v in verbs)


def is_modify_request(text: str) -> bool:
    return _is_request(text, MODIFY_VERBS)


def is_action_request(text: str) -> bool:
    return _is_request(text, ACTION_VERBS)


def is_run_request(text: str) -> bool:
    return _is_request(text, {"run", "execute", "install", "test", "debug"})


CHANGE_CLAIM_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|'ve\s+)?(?:updated|changed|modified|added|"
    r"removed|restored|re-enabled|fixed|repaired|improved|enhanced|"
    r"redesigned|implemented|created|wrote|saved|enabled|disabled|replaced|"
    r"adjusted|tweaked|polished|refactored)\b"
    r"|\b(?:has|have)\s+been\s+(?:updated|changed|modified|added|removed|"
    r"restored|re-enabled|fixed|repaired|improved|enhanced|redesigned|"
    r"implemented|created|saved|enabled|disabled|replaced|adjusted|tweaked|"
    r"polished|refactored)\b"
    r"|\b(?:the|this)\s+(?:file|ui|code|game|page|project|layout|"
    r"dashboard|screen|site)\b[^.\n]{0,80}\b(?:updated|changed|modified|"
    r"added|removed|restored|re-enabled|fixed|repaired|improved|enhanced|"
    r"redesigned|created|saved|enabled|disabled|replaced|adjusted|tweaked|"
    r"polished|refactored)\b"
    r"|\b(?:re-enabled|restored|successfully\s+(?:updated|changed|created|"
    r"wrote|fixed|restored|improved|implemented))\b"
    r"|\bthe\s+changes?\s+(?:restore|restored|add|added|improve|improved|"
    r"update|updated)\b",
    re.I,
)


def claims_change(text: str) -> bool:
    return bool(CHANGE_CLAIM_RE.search(text or ""))


def current_request_has_mutation(messages: list[dict]) -> bool:
    """True only after a Write/Edit succeeded for the newest text request."""
    start = 0
    for index in range(len(messages or []) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and content_to_text(
            message.get("content", "")
        ).strip():
            start = index
            break
    uses: dict[str, str] = {}
    results: dict[str, bool] = {}
    for message in (messages or [])[start:]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                uses[str(block.get("id", ""))] = str(block.get("name", "")).lower()
            elif block.get("type") == "tool_result":
                results[str(block.get("tool_use_id", ""))] = bool(
                    block.get("is_error", False)
                )
    return any(
        name in MUTATION_TOOLS and not results.get(tool_id, True)
        for tool_id, name in uses.items()
    )


def project_files(cwd: str, limit: int = 40) -> list[str]:
    if not cwd:
        return []
    try:
        names = sorted(os.listdir(cwd))
    except OSError:
        return []
    out: list[str] = []
    for name in names:
        if name.startswith(".") or name in ("node_modules", "__pycache__"):
            continue
        path = os.path.join(cwd, name)
        out.append(name + ("/" if os.path.isdir(path) else ""))
        if len(out) >= limit:
            break
    return out


def last_known_file(messages: list[dict]) -> str:
    for message in reversed(messages or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                path = (block.get("input") or {}).get("file_path")
                if isinstance(path, str) and path:
                    return path
            elif block.get("type") == "tool_result":
                match = FILE_RE.search(tool_result_text(block.get("content", "")))
                if match:
                    return match.group(1)
    match = FILE_RE.search(latest_request_text(messages))
    return match.group(1) if match else ""
CODE_EXT = {
    "html": "html", "htm": "html", "css": "css", "js": "js",
    "javascript": "js", "jsx": "jsx", "ts": "ts", "typescript": "ts",
    "tsx": "tsx", "python": "py", "py": "py", "json": "json",
    "bash": "sh", "sh": "sh", "shell": "sh", "sql": "sql", "go": "go",
    "rust": "rs", "rs": "rs", "java": "java", "c": "c", "cpp": "cpp",
    "csharp": "cs", "cs": "cs", "php": "php", "rb": "rb", "ruby": "rb",
    "xml": "xml", "svg": "svg", "md": "md", "markdown": "md",
    "yaml": "yml", "yml": "yml",
}


def extract_code_dump(blocks: list[dict]):
    """Largest substantial fenced code block in a text-only reply."""
    candidates = []
    for block in blocks or []:
        if block.get("type") != "text":
            continue
        for lang, code in CODE_FENCE_RE.findall(str(block.get("text", ""))):
            lang = (lang or "").strip().lower()
            if lang not in CODE_LANGS:
                continue
            if len(code.strip()) < 400 or code.count("\n") < 8:
                continue
            candidates.append((lang, code))
    if not candidates:
        return None
    html = [c for c in candidates if c[0] in ("html", "htm")]
    lang, code = max(html or candidates, key=lambda c: len(c[1]))
    code = code.strip("\n")
    return lang, code + "\n"


def is_creation_request(text: str) -> bool:
    low = (text or "").lower()
    if re.match(r"\s*(how|what|why|explain|describe|show me how)\b", low):
        return False
    if not any(re.search(rf"\b{v}\b", low) for v in CREATION_VERBS):
        return False
    return bool(
        re.search(
            r"\b(file|page|site|html|css|js|javascript|python|script|game|app|"
            r"component|function|program|code|project|readme|config|test|"
            r"module|class|cli|tool)\b",
            low,
        )
    )


def suggested_filename(request: str, lang: str) -> str:
    match = re.search(
        r"([\w./-]+\.(?:html?|css|js|jsx|ts|tsx|py|json|sh|go|rs|java|c|"
        r"cpp|cs|php|rb|xml|svg|md|ya?ml))\b",
        request or "",
        re.I,
    )
    if match:
        return match.group(1)
    if lang in ("html", "htm"):
        return "snake.html" if "snake" in (request or "").lower() else "index.html"
    names = {
        "css": "styles.css", "js": "script.js", "javascript": "script.js",
        "jsx": "App.jsx", "ts": "main.ts", "typescript": "main.ts",
        "tsx": "App.tsx", "python": "main.py", "py": "main.py",
        "json": "data.json", "bash": "script.sh", "sh": "script.sh",
        "shell": "script.sh", "sql": "query.sql", "go": "main.go",
        "rust": "main.rs", "rs": "main.rs", "java": "Main.java",
        "c": "main.c", "cpp": "main.cpp", "csharp": "Program.cs",
        "cs": "Program.cs", "php": "index.php", "rb": "main.rb",
        "ruby": "main.rb", "xml": "document.xml", "svg": "image.svg",
        "md": "README.md", "markdown": "README.md", "yaml": "config.yml",
        "yml": "config.yml",
    }
    ext = CODE_EXT.get(lang, lang or "txt")
    return names.get(lang, f"main.{ext}")


def task_brief(messages: list[dict]) -> str:
    brief = latest_request_text(messages)
    lines = [f"LATEST REQUEST:\n{(brief or '(none)')[:4000]}"]

    last_call, last_result = last_executed_call(messages)
    if last_call and last_result:
        name = str(last_call.get("name", ""))
        raw_input = last_call.get("input") or {}
        if name.lower() == "write":
            display = {
                "file_path": raw_input.get("file_path"),
                "content": f"<{len(str(raw_input.get('content', '')))} chars>",
            }
        elif name.lower() == "edit":
            display = {
                "file_path": raw_input.get("file_path"),
                "old_string": str(raw_input.get("old_string", ""))[:120],
                "new_string": f"<{len(str(raw_input.get('new_string', '')))} chars>",
            }
        else:
            display = raw_input
        lines.append(
            "LAST EXECUTED TOOL CALL (do NOT repeat it):\n"
            f"{name} " + clip_text(json.dumps(display, ensure_ascii=False), 800)
        )
        lines.append(
            "RESULT OF THAT CALL ("
            + ("error" if last_result.get("is_error") else "ok")
            + "):\n"
            + tool_result_text(last_result.get("content", ""))[:900]
        )
        lines.append(
            "NEXT STEP: that call already ran. Do not repeat it and do not "
            "write the same file again. If it completed the request, reply "
            "now with a short final summary of what changed and where."
        )

    results = latest_tool_results(messages)
    if results:
        lines.append("LATEST TOOL RESULT(S), already executed:")
        lines.extend(results)
    return "\n\n".join(lines)


def is_duplicate_call(call: dict, messages: list[dict]) -> bool:
    last_call, last_result = last_executed_call(messages)
    if not last_call or not last_result:
        return False
    if call.get("name") != last_call.get("name"):
        return False
    return (call.get("input") or {}) == (last_call.get("input") or {})


def build_context(system: Any, tools: list[dict], messages: list[dict]) -> str:
    sys_text = clip_text(system_text(system).strip(), 26000)
    tool_text = clip_text(
        json.dumps(
            [compact_tool(t) for t in (tools or [])], ensure_ascii=False, indent=1
        ),
        24000,
    )
    transcript = serialize_messages(messages)
    sections = []
    if sys_text:
        sections.append("=== SYSTEM (from Claude Code) ===\n" + sys_text)
    if tool_text and tools:
        sections.append("=== TOOLS ===\n" + tool_text)
    if transcript:
        sections.append("=== CONVERSATION ===\n" + transcript)
    sections.append("=== END CONTEXT ===")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


class ReplyError(ValueError):
    pass


def balanced_slice(text: str, start: int) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def json_candidates(text: str) -> Iterable[str]:
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        yield match.group(1).strip()
    stripped = text.strip()
    yield stripped
    marker = re.compile(
        r'\{\s*"(?:tool_calls?|content|type|name|tool|tool_name|__tool_call__)"'
    )
    for match in marker.finditer(text):
        yield balanced_slice(text, match.start())


TOOL_FIELDS = {
    "write": {"file_path", "content"},
    "edit": {"file_path", "old_string", "new_string", "replace_all"},
    "read": {"file_path", "offset", "limit"},
    "bash": {"command", "timeout", "description", "run_in_background"},
    "glob": {"pattern", "path"},
    "grep": {
        "pattern", "path", "glob", "output_mode", "-A", "-B", "-C", "-i",
        "-n", "type", "head_limit", "multiline",
    },
    "todowrite": {"todos"},
    "webfetch": {"url", "prompt"},
    "websearch": {"query", "allowed_domains", "blocked_domains"},
    "notebookedit": {
        "notebook_path", "cell_id", "new_source", "cell_type", "edit_mode",
    },
}
def nfkc_value(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value)
    if isinstance(value, list):
        return [nfkc_value(v) for v in value]
    if isinstance(value, dict):
        return {nfkc_value(k): nfkc_value(v) for k, v in value.items()}
    return value


TOOL_ALIASES = {
    "write": {"path": "file_path", "text": "content", "code": "content"},
    "edit": {"path": "file_path", "old_text": "old_string", "new_text": "new_string"},
    "read": {"path": "file_path"},
    "bash": {"cmd": "command", "shell_command": "command"},
    "glob": {"glob_pattern": "pattern", "dir": "path"},
}


def normalize_tool_input(name: str, raw: Any) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    name_lower = (name or "").lower()
    for old_key, new_key in TOOL_ALIASES.get(name_lower, {}).items():
        if new_key not in raw and old_key in raw:
            raw[new_key] = raw.pop(old_key)
    fields = TOOL_FIELDS.get(name_lower)
    if fields is not None:
        raw = {k: v for k, v in raw.items() if k in fields}
    return nfkc_value(raw)


def salvage_bare_input(text: str, valid_tools: set[str]) -> list[dict]:
    """Turn a bare {"file_path": ...} fragment into the intended tool call."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[A-Za-z0-9_+.-]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        obj = json.loads(candidate)
    except Exception:
        obj = None
    if isinstance(obj, dict) and "file_path" in obj:
        if any(k in obj for k in ("content", "text", "code")):
            name = "Write"
        elif any(k in obj for k in ("old_string", "new_string", "old_text", "new_text")):
            name = "Edit"
        else:
            name = "Read"
        if name not in valid_tools:
            return []
        return [{"name": name, "input": normalize_tool_input(name, obj)}]

    # Bare Bash input, e.g. {"command":"echo hi"}. Only for simple commands;
    # a heredoc/redirection is likely an attempt to create a file and must go
    # through Write so it can be validated and backed up.
    if isinstance(obj, dict) and "command" in obj and "Bash" in valid_tools:
        command = str(obj["command"])
        if command.strip() and not re.search(r"(^|[^<])>{1,2}|<<", command):
            return [{"name": "Bash", "input": {"command": command}}]

    # Bare Bash fragment with unescaped quotes/newlines.
    if "Bash" in valid_tools and re.search(r'"command"\s*:\s*"', candidate):
        command_m = re.search(r'"command"\s*:\s*"', candidate)
        end = candidate.rfind('"')
        command = decode_json_fragment(candidate[command_m.end():end])
        if (
            command.strip()
            and not re.search(r"(^|[^<])>{1,2}|<<", command)
        ):
            return [{"name": "Bash", "input": {"command": command}}]

    # Bare Edit input with unescaped quotes/newlines.
    if (
        "Edit" in valid_tools
        and '"old_string"' in candidate
        and '"new_string"' in candidate
        and candidate.rstrip().endswith('"')
    ):
        path_m = re.search(r'"(?:file_path|path)"\s*:\s*"([^"]*)"', candidate)
        old_m = re.search(r'"old_string"\s*:\s*"', candidate, re.I)
        new_m = re.search(r'"new_string"\s*:\s*"', candidate, re.I)
        if path_m and old_m and new_m and new_m.start() > old_m.end():
            old_end = candidate.rfind('"', old_m.end(), new_m.start())
            if old_end < old_m.end():
                old_end = new_m.start() - 1
            new_end = candidate.rfind('"')
            if new_end > new_m.end():
                old = decode_json_fragment(candidate[old_m.end():old_end])
                new = decode_json_fragment(candidate[new_m.end():new_end])
                if old and new:
                    return [
                        {
                            "name": "Edit",
                            "input": normalize_tool_input(
                                "Edit",
                                {
                                    "file_path": decode_json_fragment(path_m.group(1)),
                                    "old_string": old,
                                    "new_string": new,
                                },
                            ),
                        }
                    ]

    # Bare Write input whose content has unescaped quotes/newlines (common
    # when the model emits HTML directly). Only accept a complete fragment.
    if "Write" in valid_tools and candidate.rstrip().endswith('"}'):
        path_m = re.search(r'"(?:file_path|path)"\s*:\s*"([^"]*)"', candidate)
        content_m = re.search(r'"(?:content|text|code)"\s*:\s*"', candidate, re.I)
        if path_m and content_m:
            end = candidate.rfind('"')
            content = decode_json_fragment(candidate[content_m.end():end])
            if len(content.strip()) >= 50:
                return [
                    {
                        "name": "Write",
                        "input": {
                            "file_path": decode_json_fragment(path_m.group(1)),
                            "content": content,
                        },
                    }
                ]
    return []


def parse_with_salvage(text: str, valid_tools: set[str]) -> tuple[list[dict], list[dict]]:
    blocks, calls = parse_reply(text, valid_tools)
    if not calls:
        salvaged = salvage_bare_input(text, valid_tools)
        if salvaged:
            debug("salvaged bare tool input as", salvaged[0]["name"])
            calls = salvaged
    return blocks, calls


def call_from_object(obj: dict) -> dict | None:
    name = None
    for key in ("name", "tool", "tool_name", "recipient_name"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    if not name and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        if isinstance(fn.get("name"), str):
            name = fn["name"].strip()
    if not name:
        return None
    raw_input = (
        obj.get("input")
        if "input" in obj
        else obj.get("arguments", obj.get("parameters"))
    )
    if raw_input is None:
        meta = {
            "name", "tool", "tool_name", "recipient_name", "function",
            "id", "type", "index", "tool_call_id",
        }
        raw_input = {k: v for k, v in obj.items() if k not in meta}
    return {"name": name, "input": normalize_tool_input(name, raw_input)}


def decode_json_fragment(raw: str) -> str:
    """Decode a JSON string body that may contain unescaped quotes/newlines."""
    out: list[str] = []
    i = 0
    mapping = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/"}
    while i < len(raw):
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(raw):
            out.append("\\")
            break
        nxt = raw[i + 1]
        if nxt in mapping:
            out.append(mapping[nxt])
            i += 2
        elif nxt == "u" and i + 5 < len(raw):
            try:
                out.append(chr(int(raw[i + 2:i + 6], 16)))
                i += 6
            except ValueError:
                out.append("\\")
                i += 1
        else:
            out.append("\\")
            out.append(nxt)
            i += 2
    return "".join(out)


def salvage_tool_call(text: str, valid_tools: set[str]):
    """Recover a nearly-JSON tool call with unescaped quotes in its content."""
    names = {t.lower(): t for t in valid_tools}

    end = text.rfind('"')

    if "edit" in names and re.search(r'"name"\s*:\s*"Edit"', text, re.I):
        path_m = re.search(r'"file_path"\s*:\s*"([^"]*)"', text)
        old_m = re.search(r'"old_string"\s*:\s*"', text, re.I)
        new_m = re.search(r'"new_string"\s*:\s*"', text, re.I)
        if path_m and old_m and new_m and new_m.start() > old_m.end():
            old_end = text.rfind('"', old_m.end(), new_m.start())
            if old_end < old_m.end():
                old_end = new_m.start() - 1
            new_end = text.rfind('"')
            if new_end > new_m.end():
                old = decode_json_fragment(text[old_m.end():old_end])
                new = decode_json_fragment(text[new_m.end():new_end])
                if old and new:
                    return {
                        "name": names["edit"],
                        "input": normalize_tool_input(
                            "Edit",
                            {
                                "file_path": decode_json_fragment(path_m.group(1)),
                                "old_string": old,
                                "new_string": new,
                            },
                        ),
                    }

    if "write" in names and re.search(r'"name"\s*:\s*"Write"', text, re.I):
        path_m = re.search(r'"file_path"\s*:\s*"([^"]*)"', text)
        content_m = re.search(r'"content"\s*:\s*"', text, re.I)
        if path_m and content_m and end > content_m.end():
            content = decode_json_fragment(text[content_m.end():end])
            if len(content.strip()) >= 20:
                return {
                    "name": names["write"],
                    "input": {
                        "file_path": decode_json_fragment(path_m.group(1)),
                        "content": content,
                    },
                }

    if "bash" in names and re.search(r'"name"\s*:\s*"Bash"', text, re.I):
        command_m = re.search(r'"command"\s*:\s*"', text)
        if command_m and end > command_m.end():
            command = decode_json_fragment(text[command_m.end():end])
            if command.strip():
                return {
                    "name": names["bash"],
                    "input": {"command": command},
                }
    return None


def salvage_prose_tool_call(text: str, valid_tools: set[str]):
    """Parse outputs like: Tool call: Write file_path: x.html Full content: ..."""
    names = {t.lower(): t for t in valid_tools}
    match = re.search(r"Tool call:\s*(Write|Bash|Edit)\b", text, re.I)
    if not match:
        return None
    name = names.get(match.group(1).lower())
    if not name:
        return None
    if name.lower() == "write":
        path_m = re.search(r"file_path:\s*`?([^\s`]+)`?", text)
        content_m = re.search(r"(?:Full content|content):\s*", text, re.I)
        if path_m and content_m:
            code = text[content_m.end():].strip()
            code = re.sub(r"^```[\w.+-]*\s*", "", code)
            code = re.sub(r"\s*```\s*$", "", code).strip()
            if len(code) > 50:
                return {
                    "name": name,
                    "input": {
                        "file_path": path_m.group(1).strip("`'\""),
                        "content": code + "\n",
                    },
                }
    if name.lower() == "bash":
        command_m = re.search(r"command:\s*`?([^`\n]+)`?", text, re.I)
        if command_m:
            return {"name": name, "input": {"command": command_m.group(1).strip()}}
    return None


def parse_reply(raw: str, valid_tools: set[str]) -> tuple[list[dict], list[dict]]:
    """Returns (content_blocks, tool_calls) for the model reply."""
    text = (raw or "").strip()
    if not text:
        return [], []

    text = re.sub(r"^\s*⏺\s*", "", text)

    for candidate in json_candidates(text):
        if not candidate:
            continue
        cleaned = candidate.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = (
            cleaned.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2019", "'")
        )
        try:
            obj = json.loads(cleaned)
        except Exception:
            continue

        blocks: list[dict] = []
        calls: list[dict] = []

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    call = call_from_object(item)
                    if call:
                        calls.append(call)
        elif isinstance(obj, dict):
            if isinstance(obj.get("content"), list):
                for block in obj["content"]:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        value = str(block.get("text", "")).strip()
                        if value:
                            blocks.append({"type": "text", "text": value})
                    elif block.get("type") == "tool_use":
                        call = call_from_object(block)
                        if call:
                            call["_id"] = block.get("id")
                            calls.append(call)
            if isinstance(obj.get("tool_calls"), list):
                for item in obj["tool_calls"]:
                    if isinstance(item, dict):
                        call = call_from_object(item)
                        if call:
                            calls.append(call)
            for key in ("tool_call", "__tool_call__"):
                value = obj.get(key)
                if isinstance(value, dict):
                    call = call_from_object(value)
                    if call:
                        calls.append(call)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            call = call_from_object(item)
                            if call:
                                calls.append(call)
            if not calls:
                call = call_from_object(obj)
                if call:
                    calls.append(call)

        if not calls and not blocks:
            continue

        # Drop unknown tool names; remember them for the repair message.
        known = {t.lower(): t for t in valid_tools}
        resolved = []
        unknown = []
        for call in calls:
            name = call["name"]
            if name in valid_tools:
                resolved.append(call)
            elif name.lower() in known:
                call["name"] = known[name.lower()]
                resolved.append(call)
            else:
                unknown.append(name)

        if resolved:
            return blocks, resolved
        if unknown:
            raise ReplyError(
                "unknown tool(s): " + ", ".join(sorted(set(unknown)))
            )

    # Nearly-JSON tool call with unescaped quotes/newlines inside a string.
    salvaged = salvage_tool_call(text, valid_tools)
    if salvaged:
        return [], [salvaged]

    # Prose form: "Tool call: Write file_path: ... Full content: ...".
    salvaged = salvage_prose_tool_call(text, valid_tools)
    if salvaged:
        return [], [salvaged]

    # No JSON at all: plain text answer.
    return [{"type": "text", "text": text}], []


def looks_like_tool_attempt(raw: str) -> bool:
    lowered = (raw or "").lower()
    return any(
        marker in lowered
        for marker in ('"tool_calls"', '"tool_call"', '"tool_use"', '"name"', '"tool"')
    ) and "{" in raw


# ---------------------------------------------------------------------------
# Browser automation
# ---------------------------------------------------------------------------

AGE_MARKERS = ("confirm your age", "13 years of age or older", "younger than 13")
CONSENT_MARKERS = ("accept & continue", "by continuing, you agree")


def page_text(driver) -> str:
    try:
        return driver.find_element(By.TAG_NAME, "body").text or ""
    except Exception:
        return ""


def find_textarea(driver):
    for selector in (
        "textarea.cri-textarea",
        "textarea[placeholder]",
        "textarea",
        "div[contenteditable='true']",
    ):
        try:
            for element in driver.find_elements(By.CSS_SELECTOR, selector):
                if element.is_displayed() and element.is_enabled():
                    return element
        except Exception:
            continue
    return None


def click_text(driver, predicate) -> bool:
    xpath = (
        "//*[self::button or self::a or self::label or self::input "
        "or @role='button' or @role='checkbox' or @role='radio']"
    )
    try:
        elements = driver.find_elements(By.XPATH, xpath)
    except Exception:
        return False
    for element in elements:
        try:
            if not element.is_displayed() or not element.is_enabled():
                continue
            text = " ".join(
                part
                for part in (
                    (element.text or "").strip(),
                    (element.get_attribute("aria-label") or "").strip(),
                    (element.get_attribute("value") or "").strip(),
                )
                if part
            ).lower()
            if text and predicate(text):
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});", element
                )
                driver.execute_script("arguments[0].click();", element)
                return True
        except Exception:
            continue
    return False


def is_age_gate(driver) -> bool:
    body = page_text(driver).lower()
    if "confirm your age" in body:
        return True
    return "13 years of age or older" in body and "younger than 13" in body


def is_consent_gate(driver) -> bool:
    body = page_text(driver).lower()
    if "accept & continue" in body:
        return True
    return "by continuing, you agree" in body and "privacy policy" in body


def click_any_checkbox(driver) -> bool:
    """Fallback for age gates whose checkbox has no accessible text."""
    selectors = (
        "input[type='checkbox']",
        "[role='checkbox']",
        "input[type='radio']",
        "[role='radio']",
    )
    for selector in selectors:
        try:
            for element in driver.find_elements(By.CSS_SELECTOR, selector):
                try:
                    if not element.is_displayed() or not element.is_enabled():
                        continue
                    checked = False
                    try:
                        checked = element.is_selected()
                    except Exception:
                        checked = (
                            (element.get_attribute("aria-checked") or "").lower()
                            == "true"
                        )
                    if checked:
                        continue
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});", element
                    )
                    driver.execute_script("arguments[0].click();", element)
                    return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


def dismiss_gates(driver, timeout: float = 12.0) -> bool:
    """
    Clear the aifreeforever "Please confirm your age to chat with AI" gate
    (13+ confirmation + Continue) and the following terms/ad consent gate
    ("Accept & Continue"). Never clicks the "younger than 13" option.
    """
    if not (is_age_gate(driver) or is_consent_gate(driver)):
        return True

    log("age/consent gate detected; accepting")
    deadline = time.time() + timeout

    while time.time() < deadline:
        if is_age_gate(driver):
            # Tick the real checkbox first; clicking Continue before the
            # state updates leaves the gate stuck on some page builds.
            clicked = click_any_checkbox(driver)
            if not clicked:
                clicked = click_text(
                    driver,
                    lambda t: "13 years of age or older" in t
                    or ("confirm" in t and "13" in t and "younger" not in t)
                    or ("i am" in t and "13" in t and "younger" not in t),
                )

            # Let React enable Continue after the checkbox tick, then click it.
            time.sleep(0.8)
            click_text(
                driver, lambda t: t in {"continue", "confirm", "agree", "proceed"}
            )
            time.sleep(0.8)
            continue

        if is_consent_gate(driver):
            if click_text(
                driver,
                lambda t: ("accept" in t and "continue" in t)
                or t in {"accept", "continue", "agree & continue"},
            ):
                time.sleep(0.5)
                continue

        log("age/consent gate cleared")
        return True

    still = is_age_gate(driver) or is_consent_gate(driver)
    if still:
        time.sleep(1.5)
        still = is_age_gate(driver) or is_consent_gate(driver)
    if still:
        log("age/consent gate STILL visible after confirmation attempts")
    return not still


def safe_get(driver, url: str) -> None:
    try:
        driver.get(url)
    except TimeoutException:
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass


def profile_processes() -> list[int]:
    """PIDs of Chrome/chromedriver processes using this bridge's profile."""
    pids: list[int] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            comm = open(f"/proc/{pid}/comm", encoding="utf-8").read().strip()
            cmd = (
                open(f"/proc/{pid}/cmdline", "rb")
                .read()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except Exception:
            continue
        if comm in ("chrome", "chromedriver") and PROFILE_DIR in cmd:
            pids.append(int(pid))
    return pids


def clear_stale_profile() -> None:
    """Kill orphaned bridge browsers and clear Chrome's profile locks.

    Safe: only processes whose command line contains this bridge's dedicated
    profile directory are touched.
    """
    pids = profile_processes()
    if pids:
        log("cleaning stale browser processes:", " ".join(map(str, pids)))
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        deadline = time.time() + 4
        while time.time() < deadline and profile_processes():
            time.sleep(0.2)
        for pid in profile_processes():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(0.3)
    for name in ("SingletonCookie", "SingletonLock", "SingletonSocket"):
        try:
            os.remove(os.path.join(PROFILE_DIR, name))
        except FileNotFoundError:
            pass
        except Exception:
            pass


def make_driver():
    options = Options()
    options.binary_location = CHROME_BINARY
    if HEADLESS:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-minimized")
    options.page_load_strategy = "eager"
    options.add_argument(f"--user-data-dir={PROFILE_DIR}")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--js-flags=--max-old-space-size=2048")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER), options=options)
    driver.set_page_load_timeout(60)
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": (
                    "Object.defineProperty(navigator, 'webdriver', "
                    "{get: () => undefined});"
                )
            },
        )
    except Exception:
        pass
    return driver


ASSISTANT_SELECTOR = "div.cri-assistant"
STREAMING_SELECTOR = ".cri-assistant.is-streaming"
THINKING = ("ai is thinking", "thinking...", "thinking…", "generating...")


def is_thinking(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    return value in THINKING or value.startswith("ai is thinking")


def assistant_elements(driver):
    try:
        return driver.find_elements(By.CSS_SELECTOR, ASSISTANT_SELECTOR)
    except Exception:
        return []


def user_message_count(driver) -> int:
    try:
        return len(driver.find_elements(By.CSS_SELECTOR, "div.cri-user"))
    except Exception:
        return 0


def message_text(element) -> str:
    try:
        inner = element.find_elements(By.CSS_SELECTOR, ".markdown-content")
        node = inner[0] if inner else element
        return (node.get_attribute("innerText") or node.text or "").strip()
    except Exception:
        return ""


def is_streaming(driver) -> bool:
    try:
        for element in driver.find_elements(By.CSS_SELECTOR, STREAMING_SELECTOR):
            if element.is_displayed():
                return True
    except Exception:
        pass
    elements = assistant_elements(driver)
    if elements:
        try:
            if elements[-1].is_displayed() and is_thinking(message_text(elements[-1])):
                return True
        except Exception:
            pass
    return False


class BrowserBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.driver = None
        self._last_error = ""

    # -- driver lifecycle ---------------------------------------------------
    def _ensure_driver(self):
        if self.driver is None:
            os.makedirs(PROFILE_DIR, exist_ok=True)
            clear_stale_profile()
            log("starting browser")
            self.driver = make_driver()
        return self.driver

    def restart(self):
        try:
            if self.driver is not None:
                self.driver.quit()
        except Exception:
            pass
        self.driver = None
        clear_stale_profile()

    def close(self):
        try:
            if self.driver is not None:
                self.driver.quit()
        except Exception:
            pass
        finally:
            self.driver = None
            clear_stale_profile()

    # -- page ---------------------------------------------------------------
    def _fresh_chat(self, driver):
        safe_get(driver, CHAT_URL)
        WebDriverWait(driver, 60).until(lambda d: find_textarea(d))
        if not dismiss_gates(driver, 20):
            safe_get(driver, CHAT_URL)
            WebDriverWait(driver, 60).until(lambda d: find_textarea(d))
            if not dismiss_gates(driver, 20):
                raise RuntimeError("age/consent gate not cleared")
        try:
            if "aifreeforever.com/chat/" not in (driver.current_url or ""):
                raise RuntimeError(f"not on chat page: {driver.current_url!r}")
        except WebDriverException:
            raise

    def _set_input(self, driver, element, text: str) -> None:
        dismiss_gates(driver)
        try:
            element.click()
        except Exception:
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});"
                    "arguments[0].focus();",
                    element,
                )
            except Exception:
                pass
        driver.execute_script("arguments[0].focus();", element)
        driver.execute_script(
            """
            const el = arguments[0];
            el.focus();
            if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
                const proto = Object.getPrototypeOf(el);
                const desc = Object.getOwnPropertyDescriptor(proto, 'value');
                if (desc && desc.set) desc.set.call(el, '');
                else el.value = '';
                el.dispatchEvent(new Event('input',{bubbles:true}));
                el.dispatchEvent(new Event('change',{bubbles:true}));
            } else {
                const sel = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(el);
                sel.removeAllRanges();
                sel.addRange(range);
            }
            """,
            element,
        )
        time.sleep(0.15)
        driver.execute_script(
            """
            const el = arguments[0];
            const value = arguments[1];
            const proto = el.tagName === 'TEXTAREA'
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype;
            const desc = Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) desc.set.call(el, value);
            else el.value = value;
            el.dispatchEvent(new Event('input',{bubbles:true}));
            el.dispatchEvent(new Event('change',{bubbles:true}));
            """,
            element,
            text,
        )
        time.sleep(0.3)
        try:
            got = driver.execute_script(
                "return arguments[0].value ? arguments[0].value.length : "
                "(arguments[0].innerText || '').length;",
                element,
            )
        except Exception:
            got = len(text)
        for _ in range(12):
            if got >= len(text) * 0.999:
                break
            time.sleep(0.25)
            try:
                got = driver.execute_script(
                    "return arguments[0].value ? arguments[0].value.length : 0;",
                    element,
                )
            except Exception:
                break
        if got < len(text) * 0.95:
            debug(f"input readback {got}/{len(text)}, retrying set")
            driver.execute_script(
                """
                const el = arguments[0];
                const proto = Object.getPrototypeOf(el);
                const desc = Object.getOwnPropertyDescriptor(proto, 'value');
                if (desc && desc.set) desc.set.call(el, arguments[1]);
                el.dispatchEvent(new Event('input',{bubbles:true}));
                """,
                element,
                text,
            )
            time.sleep(0.8)

    def _submit(self, driver) -> bool:
        dismiss_gates(driver)
        selectors = (
            "button[data-testid='send-button']",
            "button[aria-label*='Send' i]",
            "button[type='submit']",
            "button[class*='send' i]",
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            for selector in selectors:
                try:
                    for button in driver.find_elements(By.CSS_SELECTOR, selector):
                        if button.is_displayed() and button.is_enabled():
                            driver.execute_script("arguments[0].click();", button)
                            return True
                except Exception:
                    continue
            time.sleep(0.25)
        try:
            textarea = find_textarea(driver)
            if textarea:
                textarea.send_keys(Keys.ENTER)
                return True
        except Exception:
            pass
        return False

    def _wait_answer(self, driver, before_count: int, timeout: float) -> str:
        deadline = time.time() + timeout
        last = ""
        stable_since = 0.0
        while time.time() < deadline:
            elements = assistant_elements(driver)
            text = ""
            if len(elements) > before_count:
                text = message_text(elements[-1])
            if text and not is_thinking(text) and not is_streaming(driver):
                if text == last:
                    if stable_since and time.time() - stable_since >= STABLE_SECONDS:
                        return text
                else:
                    last = text
                    stable_since = time.time()
            else:
                stable_since = 0.0
            time.sleep(0.75)
        raise TimeoutError(
            f"no stable answer within {timeout:.0f}s (last text: {last[:300]!r})"
        )

    def _send(self, driver, text: str, wait: bool, timeout: float) -> str:
        if not dismiss_gates(driver, 20):
            raise RuntimeError("age/consent gate not cleared")
        element = find_textarea(driver)
        if element is None:
            raise RuntimeError("composer not found")
        before = len(assistant_elements(driver))
        users_before = user_message_count(driver)
        self._set_input(driver, element, text)
        if not self._submit(driver):
            raise RuntimeError("send button not available")

        # Confirm the click actually submitted: the user bubble appears
        # immediately, long before the assistant answer exists. Without this
        # a swallowed click only surfaces as a 300s answer timeout.
        confirm_deadline = time.time() + 12
        submitted = False
        while time.time() < confirm_deadline:
            if (
                len(assistant_elements(driver)) > before
                or user_message_count(driver) > users_before
            ):
                submitted = True
                break
            time.sleep(0.3)

        if not submitted:
            debug("submit did not register; re-checking gates and retrying once")
            dismiss_gates(driver)
            if not self._submit(driver):
                raise RuntimeError("message did not submit")
            retry_deadline = time.time() + 12
            while time.time() < retry_deadline:
                if (
                    len(assistant_elements(driver)) > before
                    or user_message_count(driver) > users_before
                ):
                    submitted = True
                    break
                time.sleep(0.3)

        if not submitted:
            raise RuntimeError("message did not submit (no user bubble appeared)")

        if not wait:
            return ""
        answer = self._wait_answer(driver, before, timeout)
        return answer

    # -- public -------------------------------------------------------------
    def ask(
        self,
        system: Any,
        tools: list[dict],
        messages: list[dict],
    ) -> str:
        with self.lock:
            last_exc = None
            for attempt in (1, 2, 3):
                try:
                    return self._ask_once(system, tools, messages)
                except RuntimeError as exc:
                    # Submit/gate failures are page-level: _ask_once reloads
                    # the chat on the next attempt. Do not close the browser
                    # (that is what caused the open/close flicker).
                    last_exc = exc
                    log(f"attempt {attempt} failed:", repr(exc)[:300])
                    time.sleep(1.5)
                except (WebDriverException, TimeoutError) as exc:
                    last_exc = exc
                    log(f"attempt {attempt} failed:", repr(exc)[:300])
                    self.restart()
                    time.sleep(2)
            raise RuntimeError(f"browser bridge failed: {last_exc}")

    def _ask_once(self, system: Any, tools: list[dict], messages: list[dict]) -> str:
        driver = self._ensure_driver()
        self._fresh_chat(driver)

        protocol = PROTOCOL
        brief = task_brief(messages)
        context = build_context(system, tools, messages)

        valid_tools = {str(t.get("name")) for t in (tools or []) if t.get("name")}

        # One unsplit message. The protocol and the latest request/result go
        # at the very top because the site's model follows top-of-message
        # instructions and ignores trailing ones on long inputs.
        tool_names = ", ".join(sorted(valid_tools)) or "(none)"
        mode = (
            "AGENT MODE ON. Available tools: "
            + tool_names
            + ". Creating, editing, running or inspecting files REQUIRES a "
            "tool call. Pasting code in chat does not create files."
            if valid_tools
            else "No tools are available for this request."
        )
        cwd = working_directory(messages)
        work = (
            "\n\nWORKING DIRECTORY: "
            + cwd
            + "\nCreate new files inside this directory with RELATIVE paths "
            "(for example snake.html). Never write into ~/.claude, session "
            "transcript folders, or */projects/* storage paths."
            if cwd
            else ""
        )
        files = project_files(cwd)
        if files:
            work += "\nPROJECT FILES (top level): " + ", ".join(files)
        hint_file = last_known_file(messages)
        if hint_file:
            work += "\nLIKELY TARGET FILE: " + hint_file
        def compose_prompt(ctx: str) -> str:
            return (
                protocol
                + "\n\n=== TASK BRIEF ===\n"
                + mode
                + work
                + "\n\n"
                + brief
                + "\n\n=== FULL CLAUDE CODE CONTEXT ===\n"
                + ctx
                + "\n\n=== END CONTEXT ===\n"
                "Respond now using the protocol at the top of this message. "
                "Reply only with the answer."
            )

        prompt = compose_prompt(context)
        if len(prompt) > MAX_PROMPT:
            original = len(prompt)
            trimmed = list(messages or [])
            for keep in (24, 16, 12, 8, 6, 4):
                if not messages:
                    break
                trimmed = list(messages[-keep:])
                context = build_context(system, tools, trimmed)
                prompt = compose_prompt(context)
                if len(prompt) <= MAX_PROMPT:
                    break
            log(
                f"prompt capped {original} -> {len(prompt)} chars "
                f"(messages {len(messages)} -> {len(trimmed)})"
            )
        log(f"single message, {len(prompt)} chars")
        raw = self._send(driver, prompt, True, RESPONSE_TIMEOUT)
        return self._finish(raw, valid_tools, messages, driver)

    def _finish(
        self,
        raw: str,
        valid_tools: set[str],
        messages: list[dict],
        driver,
    ) -> str:
        debug("raw reply:", repr(raw[:500]))

        def parse_or_repair(text: str):
            try:
                return parse_with_salvage(text, valid_tools), text
            except ReplyError as exc:
                log("reply parse issue:", exc)
                repair = (
                    "Your previous reply could not be used: "
                    f"{exc}. Reply again with ONLY one valid JSON object from "
                    "the protocol at the top of this conversation, no markdown "
                    "fences and no prose. Use one of these exact tool names: "
                    + ", ".join(sorted(valid_tools))
                )
                fixed = self._send(driver, repair, True, RESPONSE_TIMEOUT)
                return parse_with_salvage(fixed, valid_tools), fixed

        (blocks, calls), raw = parse_or_repair(raw)
        after_tool = latest_user_has_tool_result(messages)
        mutation_done = current_request_has_mutation(messages)
        claimed_change = claims_change(raw)

        def loop_call(call: dict) -> bool:
            return is_duplicate_call(call, messages) or (
                after_tool and same_path_write(call, messages)
            )

        # Safety: never let Claude Code re-execute the exact call whose result
        # is already in the transcript, even if the web model repeats it.
        if any(loop_call(call) for call in calls):
            last_call, last_result = last_executed_call(messages)
            result_text = tool_result_text(
                (last_result or {}).get("content", "")
            )[:900]
            log("model repeated the last tool call; suppressing it")
            same_file = bool(calls) and same_path_write(calls[0], messages)
            if same_file:
                path = (last_call.get("input") or {}).get("file_path", "the file")
                corrective = (
                    f"The file {path} was just written successfully. Do NOT "
                    "call Write for it again. Reply NOW with a short final "
                    "summary of the completed work (what you created and "
                    "where). No tool call."
                )
            else:
                corrective = (
                    "STOP. The last tool call ALREADY EXECUTED and must not "
                    "run again:\n"
                    f"{last_call.get('name')} "
                    + clip_text(
                        json.dumps(
                            last_call.get("input") or {}, ensure_ascii=False
                        ),
                        400,
                    )
                    + "\n\nResult ("
                    + ("error" if (last_result or {}).get("is_error") else "ok")
                    + "):\n"
                    + (result_text or "(empty)")
                    + "\n\nDo NOT repeat that call. If the result completes "
                    "the task, reply with a short final summary in plain text. "
                    "Otherwise reply with ONE JSON object for the NEXT action:\n"
                    '{"tool_calls":[{"name":"ToolName","input":{...}}]}'
                )
            fixed = self._send(driver, corrective, True, RESPONSE_TIMEOUT)
            try:
                blocks, calls = parse_with_salvage(fixed, valid_tools)
            except ReplyError:
                return fixed

            if any(loop_call(call) for call in calls):
                log("duplicate call persisted; returning text instead")
                text = "\n".join(
                    str(block.get("text", "")).strip()
                    for block in blocks
                    if block.get("type") == "text"
                ).strip()
                return text or (
                    "That tool call already executed; its result was:\n"
                    + (result_text or "(empty)")
                )
            return fixed

        # Keep file writes inside the real project directory. The web model
        # sometimes picks Claude Code's transcript directory from context.
        cwd = working_directory(messages)
        request = latest_request_text(messages)
        creation = is_creation_request(request)

        def preflight(call_list: list[dict]):
            bad = next(
                (
                    c
                    for c in call_list
                    if is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                ),
                None,
            )
            if bad:
                return None, "Refusing to overwrite the file with identical or destructive content."
            error = None
            for call in call_list:
                error = validate_call(call, cwd)
                if error:
                    break
            if not error:
                for call in call_list:
                    backup_target(call, cwd)
                return call_list, None
            log("pre-write syntax error; asking for a fix")
            corrective = (
                "STOP. Your proposed change has a syntax error:\n"
                + error
                + "\nDo NOT write it. Fix the code first. Do NOT use "
                "backticks or ${...} template literals; use string "
                "concatenation with +. Reply with ONE corrected tool call."
            )
            fixed = self._send(driver, corrective, True, RESPONSE_TIMEOUT)
            try:
                _blocks, fixed_calls = parse_with_salvage(fixed, valid_tools)
            except ReplyError:
                return None, fixed
            if fixed_calls:
                if any(
                    is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                    for c in fixed_calls
                ):
                    return None, fixed
                retry_error = None
                for call in fixed_calls:
                    retry_error = validate_call(call, cwd)
                    if retry_error:
                        break
                if retry_error:
                    return None, fixed
                for call in fixed_calls:
                    backup_target(call, cwd)
                return fixed_calls, None
            return None, fixed

        explicit_match = re.search(
            r"[\w./-]+\.(?:html?|css|js|jsx|ts|tsx|py|json|sh|go|rs|"
            r"java|c|cpp|cs|php|rb|xml|svg|md|ya?ml)\b",
            request,
            re.I,
        )
        explicit_path = bool(explicit_match)
        changed = False
        for call in calls:
            if call.get("name", "").lower() not in ("write", "edit"):
                continue
            path = (call.get("input") or {}).get("file_path")
            if not isinstance(path, str) or not path.startswith("/"):
                continue
            suspicious = bool(
                re.search(r"/(?:\.claude/)?projects/-[^/]+/", path)
            )
            if explicit_path:
                if suspicious and explicit_match:
                    target = explicit_match.group(0)
                    log(f"using requested path {target!r} instead of {path!r}")
                    call["input"]["file_path"] = target
                    changed = True
                continue
            if cwd:
                if path == cwd or path.startswith(cwd + "/"):
                    continue
            elif not suspicious:
                continue
            base = os.path.basename(path.rstrip("/"))
            if base:
                log(f"rewriting out-of-project path {path!r} -> {base!r}")
                call["input"]["file_path"] = base
                changed = True
        if calls:
            bad = next(
                (
                    c
                    for c in calls
                    if is_identical_write(c, cwd)
                    or is_destructive_write(c, cwd)
                ),
                None,
            )
            if bad:
                inp = bad.get("input") or {}
                path = inp.get("file_path", "the file")
                try:
                    full = (
                        path
                        if os.path.isabs(path)
                        else os.path.join(cwd or ".", path)
                    )
                    old_size = len(open(full, encoding="utf-8", errors="replace").read())
                except OSError:
                    old_size = 0
                new_size = len(str(inp.get("content", "")))
                log(
                    "rejecting dangerous full-file Write "
                    f"({old_size} -> {new_size} chars); asking for Edit"
                )
                corrective = (
                    "STOP. Your Write would overwrite the existing file and "
                    f"remove most of its code ({old_size} -> {new_size} chars). "
                    "Do NOT rewrite the whole file. Keep the existing game and "
                    "UI intact. Use Edit with a SMALL old_string/new_string "
                    "patch to add only the requested change. "
                    f"File: {path}. "
                    + (
                        "Add Web Audio API sound code (AudioContext, oscillator, "
                        "gain) and call it on food eaten and game over. "
                        if re.search(r"\b(?:sound|audio|music|sfx)\b", request, re.I)
                        else ""
                    )
                    + "Reply with ONE Edit JSON call."
                )
                fixed = self._send(driver, corrective, True, RESPONSE_TIMEOUT)
                try:
                    blocks, calls = parse_with_salvage(fixed, valid_tools)
                except ReplyError:
                    return fixed
                if not calls or any(
                    is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                    for c in calls
                ):
                    return fixed
            calls, failure = preflight(calls)
            if failure:
                return failure
            return json.dumps(
                {
                    "tool_calls": [
                        {"name": c["name"], "input": c.get("input") or {}}
                        for c in calls
                    ]
                },
                ensure_ascii=False,
            )

        # A mutation that leaves broken syntax must not be summarized as done.
        if after_tool and mutation_done and not calls:
            changed_path = latest_mutation_file(messages, cwd)
            syntax_error = validate_file(changed_path)
            if syntax_error:
                log("post-write syntax error; asking for a fix")
                corrective = (
                    "STOP. The file you just changed has a syntax error:\n"
                    + syntax_error
                    + "\nFix it now with a small Edit. Do NOT summarize, do "
                    "NOT claim success, and do NOT rewrite the whole file. "
                    f"File: {os.path.basename(changed_path)}. Reply with ONE "
                    "Edit JSON call."
                )
                fixed = self._send(driver, corrective, True, RESPONSE_TIMEOUT)
                try:
                    blocks, calls = parse_with_salvage(fixed, valid_tools)
                except ReplyError:
                    return fixed
                if calls:
                    if any(
                        is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                        for c in calls
                    ):
                        log("rejecting dangerous syntax-fix Write")
                        return fixed
                    calls, failure = preflight(calls)
                    if failure:
                        return failure
                    return json.dumps(
                        {
                            "tool_calls": [
                                {"name": c["name"], "input": c.get("input") or {}}
                                for c in calls
                            ]
                        },
                        ensure_ascii=False,
                    )
                return fixed

        # The web model sometimes answers a file-creation request by pasting
        # code into chat. Claude Code would show that text and create nothing.
        # Convert the substantial code block into a real Write tool call.
        if (
            not calls
            and not mutation_done
            and "Write" in valid_tools
            and is_modify_request(request)
        ):
            dump = extract_code_dump(blocks)
            if dump:
                lang, code = dump
                path = suggested_filename(request, lang)
                log(
                    f"model pasted code ({lang}, {len(code)} chars) as text; "
                    f"converting to Write {path}"
                )
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "Write",
                                "input": {"file_path": path, "content": code},
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

        # Action requests must actually touch the project. This also catches
        # "improve the UI" being answered with a fake completion message.
        needs_mutation = is_modify_request(request)
        if not after_tool and is_action_request(request):
            needs_mutation = True
        plain_question = bool(
            re.match(
                r"\s*(how|what|why|explain|describe|is|are|does|do)\b",
                request.lower(),
            )
        )
        if claimed_change and not plain_question and (
            after_tool
            or re.search(
                r"\b(?:the|this)\s+(?:file|ui|code|game|page|project|layout)\b",
                raw,
                re.I,
            )
        ):
            needs_mutation = True
        need_tool = (
            not calls
            and bool(valid_tools)
            and not mutation_done
            and needs_mutation
        )
        if need_tool:
            hint = last_known_file(messages)
            cwd_hint = working_directory(messages)
            files = project_files(cwd_hint)
            context_bits = []
            if cwd_hint:
                context_bits.append(f"Working directory: {cwd_hint}")
            if files:
                context_bits.append("Project files: " + ", ".join(files))
            if hint:
                context_bits.append(f"Target file: {hint}")
            context_text = (" ".join(context_bits) + " ") if context_bits else ""
            if is_run_request(request) and not after_tool:
                log("run request answered without a tool call; asking for Bash")
                corrective = (
                    "STOP. You replied with text but did not execute anything. "
                    "This is a command execution request. Call Bash NOW with "
                    "the command from the request. Do not answer with the "
                    "expected output yourself. "
                    + context_text
                    + "Reply with exactly this shape, no prose: "
                    '{"tool_calls":[{"name":"Bash","input":{"command":"<command>"}}]}'
                )
            elif creation:
                log("creation request answered without a tool call; asking for Write")
                corrective = (
                    "STOP. You replied with text but did not create the file. "
                    "This is a FILE-CREATION request. Call Write NOW with "
                    "file_path and the complete file content. Do NOT use a "
                    "Bash heredoc, cat, tee, or redirection to create files. "
                    "Do NOT use backticks or ${...} template literals; use "
                    "string concatenation with +. "
                    + context_text
                    + "Reply with exactly this shape, no prose: "
                    '{"tool_calls":[{"name":"Write","input":'
                    '{"file_path":"<name>","content":"<full content>"}}]}'
                )
            elif after_tool:
                log("modify request has no file change yet; asking for Edit/Write")
                corrective = (
                    "STOP. You have not successfully changed any file yet. Do "
                    "not stop with a description, a plan, or a claim that the "
                    "work is done. Apply the requested change to the actual UI "
                    "file now with Edit or Write. Do NOT ask questions and do "
                    "NOT create a plan document. "
                    + context_text
                    + "Reply with ONE JSON tool call. Write input must contain "
                    "only file_path and content; Edit only file_path, "
                    "old_string, new_string. Preserve existing functionality."
                )
            else:
                log("action request answered without a tool call; asking for a tool")
                corrective = (
                    "STOP. You replied with text but made no tool call. This "
                    "is an action request and nothing was changed. Do NOT ask "
                    "clarifying questions, do NOT write a plan, and do NOT "
                    "claim the work is done. Inspect the project and modify "
                    "the actual UI file. Use Read first (or Glob/Bash to find "
                    "the file), then Edit or Write. "
                    + context_text
                    + "Reply NOW with ONE JSON tool call. Write input must "
                    "contain only file_path and content; Edit only file_path, "
                    "old_string, new_string."
                )
            fixed = self._send(driver, corrective, True, RESPONSE_TIMEOUT)
            try:
                blocks, calls = parse_with_salvage(fixed, valid_tools)
            except ReplyError:
                return fixed
            if calls:
                if any(
                    is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                    for c in calls
                ):
                    log("rejecting dangerous corrective Write")
                    return fixed
                calls, failure = preflight(calls)
                if failure:
                    return failure
                return json.dumps(
                    {
                        "tool_calls": [
                            {"name": c["name"], "input": c.get("input") or {}}
                            for c in calls
                        ]
                    },
                    ensure_ascii=False,
                )
            if is_run_request(request) and not after_tool:
                log("run request still had no Bash call; retrying")
                retry = self._send(
                    driver,
                    "STOP. Reply with exactly one JSON object and nothing else: "
                    '{"tool_calls":[{"name":"Bash","input":{"command":"..."}}]} '
                    "Use the command from the user's request. Do not answer "
                    "with the output.",
                    True,
                    RESPONSE_TIMEOUT,
                )
                try:
                    blocks, calls = parse_with_salvage(retry, valid_tools)
                except ReplyError:
                    return retry
                if calls:
                    return json.dumps(
                        {
                            "tool_calls": [
                                {"name": c["name"], "input": c.get("input") or {}}
                                for c in calls
                            ]
                        },
                        ensure_ascii=False,
                    )
                return retry
            if creation:
                log("creation request still had no Write call; retrying")
                retry = self._send(
                    driver,
                    "STOP. You still did not create the file. Reply with "
                    "exactly one JSON object and nothing else: "
                    '{"tool_calls":[{"name":"Write","input":{"file_path":"...",'
                    '"content":"..."}}]} '
                    "Use Write, not Bash. No backticks, no ${...}; use "
                    "string concatenation.",
                    True,
                    RESPONSE_TIMEOUT,
                )
                try:
                    blocks, calls = parse_with_salvage(retry, valid_tools)
                except ReplyError:
                    return retry
                if calls:
                    if any(
                        is_identical_write(c, cwd) or is_destructive_write(c, cwd)
                        for c in calls
                    ):
                        return retry
                    calls, failure = preflight(calls)
                    if failure:
                        return failure
                    return json.dumps(
                        {
                            "tool_calls": [
                                {"name": c["name"], "input": c.get("input") or {}}
                                for c in calls
                            ]
                        },
                        ensure_ascii=False,
                    )
                return retry
            dump = extract_code_dump(blocks)
            if dump:
                lang, code = dump
                path = suggested_filename(request, lang)
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "Write",
                                "input": {"file_path": path, "content": code},
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return fixed

        return raw


BRIDGE = BrowserBridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    BRIDGE.close()


# ---------------------------------------------------------------------------
# Anthropic response helpers
# ---------------------------------------------------------------------------


def estimate_tokens(text_or_chars: Any) -> int:
    if isinstance(text_or_chars, int):
        return max(1, text_or_chars // 4)
    return max(1, len(str(text_or_chars or "")) // 4)


def build_answer(
    raw_reply: str,
    tools: list[dict],
    model: str,
    request_chars: int,
) -> dict:
    valid_tools = {str(t.get("name")) for t in (tools or []) if t.get("name")}
    try:
        blocks, calls = parse_with_salvage(raw_reply, valid_tools)
    except ReplyError as exc:
        log("final parse failed:", exc)
        blocks, calls = [{"type": "text", "text": raw_reply}], []

    content: list[dict] = []
    for block in blocks:
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            content.append({"type": "text", "text": block["text"]})

    for call in calls:
        content.append(
            {
                "type": "tool_use",
                "id": "toolu_" + uuid.uuid4().hex[:24],
                "name": call["name"],
                "input": call["input"],
            }
        )

    if not content:
        content = [{"type": "text", "text": "(empty reply from web model)"}]

    stop_reason = "tool_use" if calls else "end_turn"
    output_text = json.dumps(content, ensure_ascii=False)

    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model or "gpt-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(request_chars),
            "output_tokens": estimate_tokens(output_text),
        },
    }


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def sse_events(answer: dict) -> Iterable[bytes]:
    yield b": ping\n\n"
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": answer["id"],
                "type": "message",
                "role": "assistant",
                "model": answer["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": answer["usage"]["input_tokens"],
                    "output_tokens": 1,
                },
            },
        },
    )

    for index, block in enumerate(answer["content"]):
        if block["type"] == "text":
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            )
        else:
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"], ensure_ascii=False),
                    },
                },
            )
        yield sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": answer["stop_reason"],
                "stop_sequence": None,
            },
            "usage": {"output_tokens": answer["usage"]["output_tokens"]},
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


async def read_json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as exc:
        raise ValueError(f"invalid JSON body: {exc}")
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


app = FastAPI(title="free-claude", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "gpt-5",
        "bridge": "free-claude",
        "url": CHAT_URL,
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await read_json(request)
    except ValueError as exc:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            status_code=400,
        )
    text = (
        system_text(body.get("system"))
        + json.dumps(body.get("tools") or [])
        + serialize_messages(body.get("messages") or [])
    )
    return {"input_tokens": estimate_tokens(text)}


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await read_json(request)
    except ValueError as exc:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            status_code=400,
        )

    messages_list = body.get("messages") or []
    if not isinstance(messages_list, list) or not messages_list:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages must be a non-empty array",
                },
            },
            status_code=400,
        )

    tools = body.get("tools") or []
    model = str(body.get("model") or "gpt-5")
    stream = bool(body.get("stream"))
    request_chars = len(
        system_text(body.get("system"))
        + json.dumps(tools, ensure_ascii=False)
        + serialize_messages(messages_list)
    )

    def run_bridge() -> dict:
        raw = BRIDGE.ask(body.get("system"), tools, messages_list)
        return build_answer(raw, tools, model, request_chars)

    if not stream:
        try:
            answer = await run_in_thread(run_bridge)
        except Exception as exc:
            log("request failed:", repr(exc)[:500])
            if DEBUG:
                traceback.print_exc()
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"free-claude bridge failed: {exc}",
                    },
                },
                status_code=502,
            )
        return JSONResponse(answer)

    async def stream_body():
        try:
            answer = await run_in_thread(run_bridge)
            for event in sse_events(answer):
                yield event
        except Exception as exc:
            log("stream failed:", repr(exc)[:500])
            if DEBUG:
                traceback.print_exc()
            yield sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"free-claude bridge failed: {exc}",
                    },
                },
            )

    return StreamingResponse(
        stream_body(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
    )


async def run_in_thread(func):
    import asyncio

    return await asyncio.to_thread(func)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log("free-claude bridge")
    log("listening:", f"http://{HOST}:{PORT}")
    log("chat url :", CHAT_URL)
    log("profile  :", PROFILE_DIR)
    log("compact  :", COMPACT_TOOLS)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
