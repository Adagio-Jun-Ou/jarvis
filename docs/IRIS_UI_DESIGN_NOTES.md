# Iris UI → Jarvis: Design-Notes Integration

Source: `C:\projects\iris-ui` ("Iris — Ambient Intelligence", a compact
FastAPI + single-page app). Integrated **as design ideas**, not code —
Jarvis's subsystems already cover most of Iris's surface with more depth.
This note maps each idea to what Jarvis now has (or deliberately keeps),
so the next reader does not re-investigate the source project.

## What Iris is

A 245-line FastAPI app (`app.py`) + one HTML page that:
1. Runs a **live camera feed loop** (cv2 → JPEG → shared in-memory frame,
   0.5s refresh) and streams frames over `/ws/live` at configurable fps.
2. Captures camera/screen stills, sends them to a **vision model**, then
   optionally chains the vision output into a **second "think" model**
   (`{vision_output}` template substitution) over a WebSocket
   (`/ws/vision-stream`) — a two-model vision→reasoning pipeline.
3. Persists **observations / notifications / manual notes** in SQLite
   (`~/.iris/memory.db`) with tag search and dismissible notifications.
4. Presents everything under an **animated iris avatar** (an eye) whose
   color/pupil/glow encode system state: `idle` (purple pulse), `seeing`
   (cyan), `thinking` (amber, constricted pupil), `streaming` (green).

## What Jarvis adopted (implemented this cycle)

### 1. Live camera feed — `core/live_feed.py` + `/api/live-camera` + `/ws/live`
Iris's shared latest-frame holder + background loop, reimplemented on
Jarvis's `WebcamCapture` (with the event-loop-off-blocking fixes from the
sight-and-sound pass). **Privacy-first divergence on purpose:** Iris's
loop runs forever from import; Jarvis's `LiveCameraFeed` runs ONLY while
at least one consumer is attached — first `attach()` starts it, last
`release()` stops it and the camera LED goes off. `/ws/live` reuses the
same WS origin guard as `/ws` (browsers always send Origin; a cross-site
page must not watch the user's camera). Frames stay in memory, never on
disk.

### 2. Two-model vision chain — `POST /api/vision-chain`
Iris's `vision → think` chain, wired to Jarvis's real dual-model
architecture instead of Iris's two hardcoded chat calls:

- Stage 1 (VISION): `multimodal.vision.analyze_image()` on a one-shot
  camera capture (camera released immediately — same privacy as the
  webcam tool).
- Stage 2 (REASON): the **backend model** (`backend_llm`,
  kimi-k2.7-code) reasons over the vision output with the caller's
  `{vision_output}` template.

Observations from the chain are reported to the Prefrontal Cortex
(same shape as `/api/analyze-image`), so seeing feeds Jarvis's
cognitive loop, not just the browser.

## What Jarvis already covers (no work needed)

| Iris idea | Jarvis equivalent |
|---|---|
| SQLite observations/notes/notifications | `ConversationMemory` + `unified_memory` + `persistent_memory` + knowledge graph — strictly richer (vector + lexical search, bounded stores, integrity checks) |
| `/api/search` keyword query | `memory_hub.retrieve()` (semantic + lexical fallback) and the web UI's search paths |
| Model selection per request | `dual_model` orchestrator + `model_coordinator` task routing (config.yaml single source of truth) |
| API-key via env | `~/.jarvis/secrets/api_keys.env` loader in run.py (outside the repo, never committed) |
| Notification dismiss/ack | Butler anticipations + proactive suggestions with acceptance/dismissal tracking |

## Design ideas parked for the frontend (not implemented)

The React app (`web/src`) is the place for these; recorded so they are
not lost:

1. **State avatar** — Iris's eye encodes idle/seeing/thinking/streaming.
   Jarvis's `/api/ui/state` already exposes `thinking`, `tool_calls`,
   and client counts; a small React component could drive a Jarvis-style
   avatar from exactly those signals (map: `thinking`→amber, active
   camera consumers→cyan "seeing", streaming tokens→green, else idle
   purple pulse).
2. **Two-pane live view** — Iris's grid (feed left, observations right)
   maps onto Jarvis's VisualTab: a live camera pane next to the existing
   stream visualization.
3. **Status badges in the header** (`active` / `streaming` pulse
   animation) — Jarvis's ChatTab status dot could adopt the streaming
   pulse for token activity.

## Model pair this cycle (user-specified)

- **Primary** `glm-5.3-flash` — chat, creative, user-facing, vision
  ("flash" = fast interactive tier).
- **Backend** `kimi-k2.7-code` — code, math, analysis, refinement.
- Fallbacks: primary → `kimi-k2.7-code`, `qwen3.5:397b`; backend →
  `glm-5.3-flash`, `glm-5.1`.
- The 38 hardcoded model-id strings across `dual_model.py` /
  `model_coordinator.py` (already diverged from config: they said
  kimi-k2.5 while config said kimi-k2.6) now resolve from the live
  clients at startup — config.yaml is the single source of truth.
- Key: `OLLAMA_API_KEY` in `~/.jarvis/secrets/api_keys.env` (loaded by
  run.py; the file lives outside the repo and can never be committed).
