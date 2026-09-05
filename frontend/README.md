# NAV_TRACE dashboard (React)

Vite + React dashboard that replays a recorded `platform-agent` run: canvas
render of the level, a risk/reward telemetry chart with a synced playhead,
playback controls (play/pause, scrub, speed — also keyboard: space, ←/→),
and a status sidebar for all 4 member workstreams.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173, hot reload
```

## Build for sharing / hosting

```bash
npm run build       # outputs to dist/
npm run preview      # serve the production build locally to sanity-check it
```

`dist/` is a static site — drop it on GitHub Pages, Vercel, Netlify, or any
static host. It fetches `trace.json` at runtime, so make sure that file
ships alongside `dist/index.html` (the build already copies everything in
`public/`).

## Updating the data

Regenerate `public/trace.json` from the repo root after changing the level
or the planner:

```bash
PYTHONPATH=. python3 demo/run_demo.py
```

(Point `demo/run_demo.py`'s output path at `frontend/public/trace.json`
directly if you want one source of truth.)

## Architecture

This is a replay viewer, not a live agent loop — `orchestrator.py` runs the
Perception → Planning → Policy → Mission pipeline offline, `demo/run_demo.py`
dumps every tick to `trace.json`, and the frontend just scrubs through that
fixed array. There's no backend and no websocket; all "live" behavior (play,
scrub, speed) is a `setInterval` walking an index into an already-complete
dataset. That choice keeps the dashboard trivial to host (static files) and
decoupled from whichever module happens to be mid-refactor.

**State ownership.** `App.jsx` is the only component that owns state: the
loaded `trace`, the current `idx`, `playing`, and `speedMs`. Every other
component is a pure function of props — `CanvasViewport`, `TelemetryChart`,
and `Ticker` all derive what they draw from `(level, ticks, idx)` or the
single current `frame = ticks[idx]`. `ModulePanel` is the one exception:
it's fully static, since module build-status isn't part of the trace data.

```
src/
  main.jsx                   React root, imports index.css
  App.jsx                    owns playback state; fetches trace.json; layout
  theme.js                   shared style primitives (see below)
  components/
    CanvasViewport.jsx       level + agent render (canvas, imperative draw per frame)
    TelemetryChart.jsx       recharts risk/reward area chart + click-to-scrub
    ModulePanel.jsx          static status card for all 4 member modules
    Ticker.jsx               bottom stats strip for the current frame
```

**Data flow per frame.** `idx` changes (timer tick, scrub, arrow key) →
`App` derives `frame = ticks[idx]` → `frame` and the surrounding `ticks`
array pass down as props → `CanvasViewport` redraws its `<canvas>`
imperatively inside a `useEffect` keyed on `[idx, ticks, level]` (canvas
isn't declarative, so this is the one place React doesn't own the pixels
directly) → `TelemetryChart` and `Ticker` re-render normally as React
components.

**Styling.** Colors and fonts are CSS custom properties in `index.css`
(`--bg`, `--panel`, `--live`, `--mono`, etc.) so the palette is themeable
from one file. `theme.js` sits on top of that: it exports the handful of
*shapes* every component repeated inline (`label`, `panel`, `pillStyle(...)`,
`STAT_TONES`) so a spacing or pill-color change doesn't require editing four
components. Everything else stays as plain inline `style={{}}` objects —
deliberately no CSS-in-JS library or Tailwind, since the component tree is
small enough that indirection would cost more than it'd save.

**Accuracy of `ModulePanel`.** All 4 interfaces in `core/interfaces.py` have
real implementations in the repo (see each `memberN_*/README.md`), and
`orchestrator.py` wires them into one loop — but the recorded run in
`trace.json` was produced by Planning's A*/MCTS search alone. `ModulePanel`
distinguishes "implemented" (true for all 4) from "drove this replay" (true
only for Planning) rather than collapsing both into a single LIVE/PENDING
pill, so the dashboard doesn't imply the RL policy or mission module
generated a run they didn't.

**Defensive edges.** `Ticker` and `CanvasViewport` both guard against a
missing/out-of-range `frame` (empty trace, a scrub past the end) instead of
throwing on `frame.risk.toFixed(...)`; `App` also surfaces a `trace.json`
fetch failure as an explicit error state rather than hanging on
`LOADING TRACE…` forever.
