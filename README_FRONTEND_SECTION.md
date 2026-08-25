## Frontend (React)

`frontend/` is a real Vite + React app (canvas level render, a recharts
risk/reward telemetry chart with a synced playhead, playback controls,
keyboard shortcuts, module-status sidebar). It replaces the earlier static
HTML/canvas version.

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

See `frontend/README.md` for build/deploy instructions and how to refresh
`public/trace.json` after changing the level or planner.
