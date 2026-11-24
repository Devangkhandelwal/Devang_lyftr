# LLM Chatbot with Key Failover and Response Caching

A full stack chat application that demonstrates three things explicitly:

- **Key failover** — a pool of provider API keys is tried in order, and a
  rate limited, exhausted or failing key hands the request to the next one.
- **Response caching** — an identical conversation is answered from the
  LiteLLM cache instead of being sent to the provider again.
- **Per message observability** — every assistant reply carries the model,
  the measured latency, the token usage of that single request, which key
  answered, and how many keys had to be skipped first.

Backend: FastAPI + LiteLLM + SQLite. Frontend: React (Vite).

> **Model note.** The brief names `gemini/gemini-2.0-flash`. Google has since
> retired that model — it now returns a 404 telling you to upgrade — so the
> default is `gemini/gemini-3.1-flash-lite`, which was verified end to end
> against live Gemini. `gemini/gemini-3.5-flash` also works;
> `gemini/gemini-3.8-flash` is frequently overloaded and returns 503. The
> model is just an environment variable (`LLM_MODEL`), so switching providers
> or models needs no code change.

---

## 1. Live URLs

Not deployed yet. Fill these in after following the deployment steps in
[Deployment](#deployment):

| Service | URL |
| --- | --- |
| Frontend | `https://<your-app>.vercel.app` |
| Backend | `https://<your-service>.onrender.com` |
| Health check | `https://<your-service>.onrender.com/health` |

---

## 2. Run locally: with Docker and without

First, create your environment file and put at least two provider keys in it:

```bash
cp .env.example .env
# then edit .env and set LLM_API_KEYS=key-one,key-two
```

`.env` is gitignored. Real keys never belong in the repository.

### With Docker

```bash
make up           # docker compose up --build -d
```

- Frontend: <http://localhost:3000>
- Backend health: <http://localhost:8000/health>

The chat history lives in the named volume `app_data`, mounted at `/data`,
so it survives container restarts:

```bash
make restart                      # history is still there
curl http://localhost:8000/sessions/s-demo/messages
```

Other targets:

```bash
make logs    # follow both services
make test    # run the backend test suite inside the backend image
make down    # stop the containers, keep the data
make clean   # stop the containers AND delete the volume (history is lost)
```

> **Note on `VITE_API_URL`.** Vite inlines the backend URL into the bundle at
> build time, so changing it requires a rebuild (`make up` rebuilds), not
> just a restart.

### Without Docker

Two terminals.

**Backend**

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt

# A relative database path is easier outside Docker:
export DATABASE_URL="sqlite:///./data/app.db"
export LLM_API_KEYS="key-one,key-two"
export LLM_MODEL="gemini/gemini-3.1-flash-lite"
export ENABLE_FAILURE_SIMULATION=true
export ALLOWED_ORIGINS="http://localhost:5173,http://localhost:3000"

uvicorn app.main:app --reload --port 8000
```

The database file, its parent directory and the schema are all created
automatically on startup.

**Frontend**

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev        # http://localhost:5173
```

**Tests**

```bash
cd backend
python -m pytest -q
```

The tests mock `litellm.acompletion`, so they need no API key and make no
network call.

> **If the browser shows "Could not reach the backend".** The origin the page
> is served from must appear verbatim in `ALLOWED_ORIGINS`, and
> `http://localhost` and `http://127.0.0.1` are *different* origins to the
> browser. On Windows `localhost` may also resolve to IPv6 `::1` while
> uvicorn binds IPv4 `127.0.0.1`, so keep the two sides consistent.

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Key fallback

Keys come from one comma separated variable, `LLM_API_KEYS`. Their position
defines their public id: the first key is `key_1`, the second `key_2`, and so
on. **Only those ids are ever logged or returned — the key values never leave
the server.**

`app/llm.py` walks the pool in order. For each key it calls:

```python
litellm.acompletion(model=..., messages=..., api_key=key, num_retries=0, timeout=30)
```

`num_retries=0` matters: LiteLLM's own retry would silently retry the *same*
dead key and hide the failover that this project is meant to show.

A failure moves to the next key and increments `fallback_attempts` when it is
a rate limit (429), an exhausted quota, a provider 5xx, a dropped connection
or a timeout. When every key has failed, `AllProvidersUnavailable` is raised
and the endpoint answers **503 `{"detail": "all providers unavailable"}`**.
Nothing is written to the history.

Two deliberate choices beyond the literal brief, both for robustness:

1. **Authentication and permission errors also fail over.** A revoked or
   misconfigured key is dead for this request, but the next key may well
   work, so it would be unhelpful to fail the whole request on it.
2. **An empty reply is treated as a provider failure.** The API contract
   promises a non-empty `reply`, so a successful call that returns no text
   hands over to the next key rather than returning an empty string.

Errors that would fail identically on every key (a malformed request, for
example) are *not* retried; there is nothing to fail over to. They surface
through the global handler as 500 `{"detail": "internal error"}`.

### Caching

`litellm.cache` is a LiteLLM `Cache` object — in memory for `CACHE_TYPE=local`,
on disk for `CACHE_TYPE=disk`. No Redis and no external cache service.

**The cache key is built from the model and the full message list only**
(system prompt + history + the new message), via
`litellm.cache.get_cache_key(model=..., messages=...)`. The API key is
deliberately *not* part of it, otherwise a reply produced by `key_1` would
never be found when the same question later runs through `key_2`. There is a
test for exactly that (`test_the_cache_key_ignores_which_api_key_answered`).

The flow is: look up the key → on a hit return immediately with
`resolution_source="LLM_CACHE"`, the identical stored reply, the stored token
usage, `key_id=null` and `fallback_attempts=0`. On a miss, call the provider
and store the response.

**A request carrying `X-Simulate-Failure` skips the cache lookup.** This is
intentional, and is the one place where behaviour is driven by the header
rather than by the message: a cache hit returns *before* the failover loop
runs, so without this the simulation would quietly do nothing on any question
that had been asked before. Such a request still stores its result.

Two notes on the current LiteLLM version (1.110.3):

- **`Cache(type="disk")` no longer exists** — LiteLLM removed its built in
  disk backend. `CACHE_TYPE=disk` is therefore served by a small
  `DiskCacheBackend` (in `app/llm.py`) plugged into the normal
  `litellm.Cache` object, so LiteLLM still owns key generation and response
  serialisation. It is backed by the `diskcache` library, which stores
  everything in a local SQLite file, so this adds no external service.
- **`max_messages` is set to `None`.** LiteLLM now refuses to cache requests
  with more than four messages by default, which would silently disable
  caching as soon as a conversation grew past two turns.

### SQLite session storage

One SQLite file holds everything, created on startup from `DATABASE_URL`
(the parent directory is created if missing).

```
messages(id, session_id, role, content, metadata, created_at)
sessions(session_id, system_prompt, created_at)
```

- **The system prompt lives in `sessions`, never as a row in `messages`.**
  That is what keeps it out of the history endpoint by construction rather
  than by filtering.
- **The user row, the assistant row and the session row are written in a
  single transaction, and only after a successful generation.** A 422 or a
  503 therefore leaves no trace at all — not even an empty session, which is
  why the history endpoint still answers 404 for it.
- `metadata` stores the assistant reply's metadata block as JSON.
- Connections are short lived and opened per operation; WAL journaling keeps
  readers from blocking the writer.
- `latency_ms` is measured on the server with `time.perf_counter()` around
  the whole request and returned as an integer.
- Token counts are read from `response.usage` of **that single response**, so
  they are never a running total across the conversation.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

All commands assume the backend is on `http://localhost:8000` and that
`ENABLE_FAILURE_SIMULATION=true`.

### Health

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "keys_configured": 2, "cache_enabled": true }
```

### Cache hit: the same first message in two brand new sessions

The second call must come back as `LLM_CACHE`, with the identical reply, the
same token counts, `key_id: null` and a visibly lower `latency_ms`.

```bash
# 1st session -> LIVE_INFERENCE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-cache-1","message":"What is the capital of France?"}'

# 2nd session, same first message -> LLM_CACHE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-cache-2","message":"What is the capital of France?"}'
```

Observed against **live Gemini** (`gemini-3.1-flash-lite`):

```
1st: "reply":"The capital of France is Paris."
     "resolution_source":"LIVE_INFERENCE"  "latency_ms":3727  "input_tokens":16
     "output_tokens":7  "key_id":"key_1"   "fallback_attempts":0

2nd: "reply":"The capital of France is Paris."     <- identical
     "resolution_source":"LLM_CACHE"       "latency_ms":9     "input_tokens":16
     "output_tokens":7  "key_id":null      "fallback_attempts":0
```

That is 3727 ms down to 9 ms, about 400x faster, with the identical reply and
the identical token counts.

> **Why a repeat question inside an *existing* session is not a cache hit.**
> The cache key covers the whole message list, so once a session has history
> the list differs and the request correctly goes live. To see a cache hit,
> ask the same question as the **first** message of a new session — which is
> exactly what reloading the page gives you.

A *different* message is never served from the cache:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-cache-3","message":"Who wrote Hamlet?"}'
```

### Failover: key_1 is rate limited, key_2 answers

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-failover","message":"Which key answered this?"}'
```

Returns **200** with a real model answer and:

```json
"key_id": "key_2", "fallback_attempts": 1, "resolution_source": "LIVE_INFERENCE"
```

`key_1=500`, `key_1=502` and `key_1=503` behave the same way. The statuses
`429`, `500`, `502` and `503` are accepted; anything else in the header is
ignored rather than rejected.

### Every key down

```bash
curl -i -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-down","message":"Nobody can answer this"}'
```

```
HTTP/1.1 503 Service Unavailable
{"detail":"all providers unavailable"}
```

And nothing was written:

```bash
curl -i -s http://localhost:8000/sessions/s-down/messages   # 404 session not found
```

### Per message token usage

```bash
curl -s http://localhost:8000/sessions/s-failover/messages
```

Each assistant row carries the usage of its own turn. In a two turn session
the input tokens grow with the conversation (for example 16 then 34) because
the history is resent — they are per request numbers, never a cumulative sum.

### Validation errors

```bash
# empty / whitespace-only message -> 422
curl -i -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-422","message":"   "}'

# message longer than 4000 characters -> 422
# missing session_id, or a session_id longer than 128 characters -> 422
# malformed JSON -> 422
```

### Unknown session

```bash
curl -i -s http://localhost:8000/sessions/does-not-exist/messages
```

```
HTTP/1.1 404 Not Found
{"detail":"session not found"}
```

### In the browser

Open <http://localhost:3000>. Every page load starts a new session, so
sending the same first question twice — reloading the page in between — shows
a green `LIVE_INFERENCE` badge the first time and a blue `LLM_CACHE` badge
with a much lower latency the second time. Each reply also shows the model,
the token counts, `key_id` and `fallback_attempts`.

---

## Deployment

### Backend on Render (Docker)

The repository contains a `render.yaml` blueprint, so the service type, the
Docker build, the health check and every non-secret environment variable are
declared in code. The quickest route is **New → Blueprint**: point Render at
this repository and it reads `render.yaml`, then prompts only for the two
values marked `sync: false` (`LLM_API_KEYS` and `ALLOWED_ORIGINS`).

The manual route below is equivalent if you prefer to click through it.

1. Push this repository to GitHub.
2. In Render: **New → Web Service**, connect the repository.
3. Settings:
   - **Runtime:** Docker
   - **Root Directory:** `backend`
   - **Dockerfile Path:** `backend/Dockerfile`
4. **Environment variables:**

   | Key | Value |
   | --- | --- |
   | `LLM_API_KEYS` | `your-first-key,your-second-key` |
   | `LLM_MODEL` | `gemini/gemini-3.1-flash-lite` |
   | `DATABASE_URL` | `sqlite:////data/app.db` (with a disk) or `sqlite:///./data/app.db` (free tier) |
   | `ALLOWED_ORIGINS` | your frontend URL, e.g. `https://your-app.vercel.app` |
   | `ENABLE_FAILURE_SIMULATION` | `true` |
   | `DEFAULT_SYSTEM_PROMPT` | `You are a helpful, concise assistant.` |
   | `CACHE_TYPE` | `disk` |

5. **Storage.** Render's free tier does not support disks, so this
   deployment runs without one: `DATABASE_URL` points at a path inside the
   container and the SQLite file lives on ephemeral storage. History is
   kept for the life of the instance — multi-turn conversations, the
   history endpoint and the cache all behave normally — but it is cleared
   when Render redeploys or wakes the service from sleep.

   To make history durable, move to a paid instance and add a disk
   (Settings → Disks) with mount path `/data`, 1 GB, then set
   `DATABASE_URL=sqlite:////data/app.db`. The image already creates `/data`
   owned by the non-root `appuser`, so no extra permission setup is needed.
   Locally, `docker compose` always uses the named volume `app_data`, so
   `make up` keeps history across restarts regardless.
6. Deploy, then check `https://<service>.onrender.com/health`.

> On Render's free tier the service sleeps when idle, so the first request
> after a pause is slow. That is the platform waking up, not the model.

### Frontend on Vercel

1. **New Project**, import the same repository.
2. Settings:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
3. **Environment variable:** `VITE_API_URL = https://<service>.onrender.com`
   (no trailing slash).
4. Deploy, then copy the resulting Vercel URL back into the backend's
   `ALLOWED_ORIGINS` and redeploy the backend. The browser blocks the call
   otherwise, and the UI will show "Could not reach the backend".

`VITE_API_URL` is baked in at build time, so changing it needs a redeploy of
the frontend, not just a restart.

---

## 5. Setup Used: your editor and AI tools

- **Editor:** Visual Studio Code on Windows 11.
- **AI tool:** Claude Code (Anthropic's CLI coding agent), model Claude Opus 5,
  driven from the integrated terminal.
- **How it was used:** the whole project was built from a single phased
  specification. Rather than writing the LiteLLM cache integration from
  memory, the agent first inspected the installed library's actual source
  (`Cache.get_cache_key`, `get_cache`, `add_cache`, `should_use_cache`) and
  verified behaviour with throwaway probe scripts. That is how the two
  version specific issues documented above — the removed `disk` backend and
  the new `max_messages=4` default — were found before any application code
  depended on them.
- **Verified in this environment:** `pytest` (46 passing), the full HTTP
  contract through `curl` against a running server, the disk cache surviving
  a process restart, the production Vite build, and the complete UI in Chrome
  (both badge states, multi-turn memory, the error bubble, every
  `data-testid`).
- **Verified against live Gemini:** a real API key was used for an end to end
  run. Live inference returned a genuine answer in 3727 ms with real token
  usage from Gemini's own `usage` block; the identical question in a new
  session came back from the cache in 9 ms. Simulated `key_1=429` and
  `key_1=500` both failed over to `key_2` with real model answers, `*=429`
  returned 503 with nothing written, and a two turn conversation correctly
  recalled earlier details with per request token counts (16 then 51 input
  tokens, not a cumulative sum). The same was then confirmed in the browser.
- **Real failover was also observed unprompted.** On two occasions Gemini
  returned a genuine transient error on `key_1` and the request completed on
  `key_2` with `fallback_attempts: 1` — the failover path working against a
  real provider outage rather than a simulated one.
- **Not verified here:** anything requiring Docker. Docker Desktop needs WSL2
  on Windows 11 Home, which requires an administrator elevation and a reboot,
  so `make up` was not executed. The Dockerfiles, `docker-compose.yml` and
  `Makefile` were written and statically checked (valid YAML, tab indented
  recipes, the healthcheck command run by hand against a live server), but
  they have not been built. Build them once before relying on them.

---

## 6. Bonus features (if any)

- **`key_id` and `fallback_attempts` shown in the UI** under every assistant
  reply, next to the required metadata, as `data-testid="meta-key-id"` and
  `data-testid="meta-fallback-attempts"`.
- **Error bubbles carry the server's `detail`.** A 422's field errors are
  flattened into a readable sentence instead of raw JSON, and the bubble has
  its own `data-testid="error-message"` so it is never mistaken for an
  assistant reply with missing metadata.
- **Cross key cache reuse is tested explicitly** — a reply produced by
  `key_2` during a simulated `key_1` outage is served from the cache on the
  next identical question.
- **Persistent response cache.** `CACHE_TYPE=disk` keeps the cache across
  restarts, verified by replaying an identical question in a fresh process
  and getting `LLM_CACHE` with zero provider calls.
- **Empty replies never reach the client** — they are treated as a provider
  failure and fail over to the next key.
- **Secrets are tested, not just promised.** Tests assert that no response
  body or health payload ever contains a key value, and the failover logger
  prints only the public key id.
- **Extra `data-testid`s for convenience:** `user-message`, `session-id`,
  `loading`.
- **Quality of life:** auto-scroll, Enter to send with Shift+Enter for a
  newline, input disabled while a request is in flight and refocused
  afterwards so you can keep typing, and WAL mode on SQLite.
