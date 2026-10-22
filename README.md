# free-claude

A local [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) bridge that lets **Claude Code** use the **aifreeforever GPT-5 web chat** as its model backend.

```text
Claude Code  ->  http://127.0.0.1:8000  ->  Chrome  ->  aifreeforever GPT-5
```

The bridge accepts normal Anthropic API requests, drives a real Chrome session, converts the web model's reply into Anthropic content blocks or server-sent events, and lets Claude Code execute the resulting tool calls.

> **Unofficial project.** Not affiliated with Anthropic, OpenAI, or aifreeforever. It relies on browser automation against a third-party website, so it can break when that site changes. Use it at your own risk and respect the site's terms and rate limits.

---

## Features

- Anthropic-compatible `POST /v1/messages` endpoint with `stream: true` and `stream: false`.
- `POST /v1/messages/count_tokens` and `GET /health`.
- Full Claude Code tool loop: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, and other tools exposed by the client.
- One fresh browser chat per API request, with the full Claude Code transcript replayed.
- Headless Chrome by default.
- Persistent Chrome profile, so login and consent state survive restarts.
- Parses several reply formats and salvages malformed tool JSON.
- Pre-write syntax validation for HTML/JS/Python/JSON, with corrective retries.
- Automatic backups before file mutations.
- Guards against identical and destructive full-file rewrites.
- NFKC normalization for Unicode math characters produced by the web UI.

---

## Requirements

- **Python 3.10+**
- **Google Chrome**
- **ChromeDriver** matching the installed Chrome major version
- **Claude Code CLI**
- **Node.js** (used for `node --check` syntax validation; optional but
  recommended, and already required by Claude Code)
- Python packages:
  - `fastapi`
  - `uvicorn`
  - `selenium`
- Internet access to `https://aifreeforever.com/chat/gpt-5`
- Linux is the tested platform. macOS should work with the Chrome/ChromeDriver paths adjusted; Windows is untested.

Install Claude Code if needed:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

---

## Installation

### 1. Get the project

```bash
git clone <your-repo-url> free-claude
cd free-claude
```

Or copy this directory to the machine that will run the bridge.

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not want to use `requirements.txt`:

```bash
pip install "fastapi>=0.110" "uvicorn>=0.29" "selenium>=4.20"
```

### 3. Install Chrome and ChromeDriver

Check the versions:

```bash
google-chrome --version
chromedriver --version
```

The **major** versions must match (for example, Chrome `152.x` needs ChromeDriver `152.x`). If ChromeDriver is not already installed, download it from [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/).

The bridge defaults to:

```text
Chrome:       /usr/bin/google-chrome
ChromeDriver: /usr/local/bin/chromedriver-152
```

Override them if your paths differ:

```bash
export FREE_CLAUDE_CHROME="$(command -v google-chrome)"
export FREE_CLAUDE_CHROMEDRIVER="$(command -v chromedriver)"
```

### 4. Start the bridge

```bash
python3 free-claude.py
```

You should see:

```text
[free-claude ...] listening: http://127.0.0.1:8000
[free-claude ...] chat url : https://aifreeforever.com/chat/gpt-5
```

Leave this process running.

On the first run, the bridge creates a Chrome profile at `~/.config/free-claude`. If the site shows an age/consent gate, the bridge tries to accept it automatically. If it cannot, run once with a visible browser:

```bash
FREE_CLAUDE_HEADLESS=0 python3 free-claude.py
```

Complete the gate in the browser, stop the bridge with `Ctrl+C`, then start it again headless.

### 5. Point Claude Code at the bridge

Use environment variables:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=gpt-5

claude
```

Or add this to `~/.claude/settings.json` (merge with any existing settings):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "gpt-5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5"
  }
}
```

The bridge does **not** validate `ANTHROPIC_AUTH_TOKEN`; it can be any non-empty string.

---

## Usage

### Interactive Claude Code

```bash
cd ~/projects/my-app
claude
```

Claude Code will send its system prompt, tool definitions, conversation, and tool results to the local bridge. The bridge forwards them to the web chat and returns tool calls, which Claude Code executes normally.

### One-shot prompt

```bash
claude -p "create a snake game in snake.html"
```

### Health check

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expected:

```json
{
  "status": "ok",
  "model": "gpt-5",
  "bridge": "free-claude",
  "url": "https://aifreeforever.com/chat/gpt-5"
}
```

### Direct API request

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-5",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

For streaming, add `"stream": true`; the response uses Anthropic-style SSE events.

---

## Configuration

All configuration is through environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `FREE_CLAUDE_HOST` | `127.0.0.1` | Address to bind. Keep it local. |
| `FREE_CLAUDE_PORT` | `8000` | HTTP port. |
| `FREE_CLAUDE_URL` | `https://aifreeforever.com/chat/gpt-5` | Chat page to automate. |
| `FREE_CLAUDE_PROFILE` | `~/.config/free-claude` | Persistent Chrome profile directory. |
| `FREE_CLAUDE_HEADLESS` | `1` | `0` shows the browser window. |
| `FREE_CLAUDE_COMPACT_TOOLS` | `1` | `0` sends tool schemas verbatim instead of compacting them. |
| `FREE_CLAUDE_TIMEOUT` | `300` | Seconds to wait for one web reply. |
| `FREE_CLAUDE_MAX_PROMPT` | `70000` | Maximum prompt characters sent to the site. |
| `FREE_CLAUDE_CHROME` | `/usr/bin/google-chrome` | Chrome executable. |
| `FREE_CLAUDE_CHROMEDRIVER` | `/usr/local/bin/chromedriver-152` | ChromeDriver executable. |
| `FREE_CLAUDE_DEBUG` | `0` | `1` enables verbose logging, including raw replies. |

Example with a different port and visible browser:

```bash
FREE_CLAUDE_PORT=8010 FREE_CLAUDE_HEADLESS=0 FREE_CLAUDE_DEBUG=1 \
  python3 free-claude.py
```

Then set `ANTHROPIC_BASE_URL=http://127.0.0.1:8010`.

---

## How it works

1. Claude Code sends a normal Anthropic Messages request to the local server.
2. The bridge flattens the system prompt, tool schemas, messages, and tool results into one self-contained prompt.
3. It opens a fresh chat in Chrome, submits the prompt, and waits for the reply to stabilize.
4. The reply is parsed as either plain text or a `tool_calls` JSON object.
5. The bridge converts that into Anthropic `text` and `tool_use` content blocks, or Anthropic SSE events.
6. Claude Code executes the tool calls and sends the next request with the results. The loop repeats until the model returns a final text answer.

The browser conversation is not continued between API calls. Every request starts from a reloaded page, which avoids stale state and keeps the behavior close to a stateless API.

---

## Reliability and safety guards

The web UI is not a perfect API. The bridge includes several protections:

- **Syntax validation before writes.** Writes and edits to HTML, JavaScript, Python, JSON, and shell files are checked before execution. A change that would introduce a syntax error is refused and sent back to the model for correction.
- **Backups.** Before a file is mutated, a copy is written to `.free-claude-backups/` inside the working project (the newest 30 are kept).
- **Destructive-write guard.** A full-file `Write` that shrinks a file by more than 30%, or removes expected game/app markers, is rejected.
- **Identical-write guard.** Re-writing a file with exactly the same content is rejected to avoid loops.
- **Unicode normalization.** The site sometimes renders `${...}` as Unicode math italic characters; tool inputs are NFKC-normalized before execution.
- **Malformed JSON salvage.** Bare `file_path`, `old_string`, `new_string`, and similar fragments are recovered when the model omits the tool wrapper.
- **No template literals in generated JS.** The bridge instructs the model to use string concatenation because the site strips backticks.

These are best-effort protections. Keep backups and review important changes.

---

## Troubleshooting

### `Address already in use`

Use another port:

```bash
FREE_CLAUDE_PORT=8010 python3 free-claude.py
```

Then update `ANTHROPIC_BASE_URL`.

### ChromeDriver version mismatch

Typical error:

```text
SessionNotCreatedException: This version of ChromeDriver only supports Chrome version ...
```

Install a ChromeDriver that matches `google-chrome --version`, then set:

```bash
export FREE_CLAUDE_CHROMEDRIVER=/path/to/matching/chromedriver
```

### Chrome fails to start / profile is locked

Close any Chrome instances using the bridge profile, then remove stale lock files:

```bash
rm -f ~/.config/free-claude/SingletonLock
rm -f ~/.config/free-claude/SingletonCookie
rm -f ~/.config/free-claude/SingletonSocket
```

Or point the bridge at a fresh profile:

```bash
FREE_CLAUDE_PROFILE=~/.config/free-claude-test python3 free-claude.py
```

### Age or consent gate keeps appearing

Run once with `FREE_CLAUDE_HEADLESS=0`, complete the gate manually, and confirm the profile persists. Do not delete `FREE_CLAUDE_PROFILE` between runs.

### Requests time out or return 502

- Check the bridge terminal output.
- Run with `FREE_CLAUDE_DEBUG=1` for raw replies.
- Increase `FREE_CLAUDE_TIMEOUT`.
- The upstream site may be slow, rate-limiting, or showing a new gate.
- Restart the bridge; if the browser is wedged, stop the process and start it again.

### Claude Code gets text instead of tool calls

- Start a fresh Claude Code session instead of resuming a very long one; the bridge caps prompts at `FREE_CLAUDE_MAX_PROMPT`.
- Run with `FREE_CLAUDE_DEBUG=1` and inspect the raw reply.
- The web model sometimes ignores the tool protocol on long or messy transcripts.

### A file was corrupted or overwritten

Check `.free-claude-backups/` in the project directory. The backup filenames include a timestamp and the original filename.

---

## Files

```text
free-claude.py      HTTP server, Chrome automation, prompt/response bridge
requirements.txt    Python dependencies
README.md           This file
```

Runtime artifacts:

```text
~/.config/free-claude/          Persistent Chrome profile
<project>/.free-claude-backups/ Per-project file backups
```

---

## Security notes

- The bridge has **no authentication**. Keep it bound to `127.0.0.1` and never expose it to a network or the internet.
- The Chrome profile contains cookies and session state for the upstream site. Treat it as sensitive.
- The bridge can execute shell commands through Claude Code's tools. Run it only in workspaces where you are comfortable with an autonomous coding agent.
- This is a development tool, not a production API gateway. There is no SLA, rate limiting, or multi-tenant isolation.

---

## Limitations

- Browser automation is fragile; a DOM or layout change on the upstream site can break the bridge.
- The upstream web model is slower than a direct API and may be rate-limited.
- Long Claude Code sessions are trimmed to fit `FREE_CLAUDE_MAX_PROMPT`; very old context can be lost.
- The bridge does not implement every Anthropic API feature. `max_tokens`, `temperature`, and similar sampling fields are accepted but not meaningfully enforced.
- Only one Chrome/browser session is used, so requests are effectively serialized.
- The upstream site may change its markup, consent flow, or model names at any time.

---

## Disclaimer

This project is for personal development and experimentation. It is not endorsed by or affiliated with Anthropic, OpenAI, or aifreeforever. You are responsible for complying with the terms of service of any website you automate and for the code executed by the agent.
