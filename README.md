# Intelligent Platform Navigation Agent

Autonomous agent for platform-game levels (gaps, moving platforms, enemies,
collectibles). Four independent workstreams, one shared contract.

**→ Open `frontend/index.html` in a browser to watch a real recorded run** —
the agent using the working A* + MCTS planner, navigating a hand-built level,
timing a jump onto a moving platform, dodging a patrolling enemy, and
collecting an item on the way to the goal.

## Status at a glance

| # | Package | Owns | Status |
|---|---|---|---|
| 1 | `member1_perception/` | CNN visual perception + LSTM temporal modeling | ✅ **Implemented** |
| 2 | `member2_planning/` | A* global route planning + MCTS tactical decisions | ✅ **Implemented** |
| 3 | `member3_rl/` | PPO/DQN policy learning, generalization | 🔲 Stub — interface only |
| 4 | `member4_llm_rag/` | LLM/SLM + RAG mission interpretation, integration | 🔲 Stub — interface only (orchestrator wiring exists) |

Planning (#2) is real and working end to end — not a stub — proven against a
hand-built demo level (`demo/`) with a rendered dashboard (`frontend/`) as
evidence. This exists to unblock the other three: it shows the shared
contract actually produces something, gives everyone a concrete `Plan`
object to build against, and gives the demo env something believable to
run. The other three packages are exactly as scaffolded in the original
commit — READMEs, stub classes, milestones — untouched, so nobody's work is
overwritten.

## Architecture

```
        MissionGuidance (periodic, LLM/RAG)
                 |
Frame -> [Perception: CNN+LSTM] -> PerceptionOutput --\
                                                         > [Policy: PPO/DQN] -> Action -> Env
GameState -> [Planning: A*+MCTS] -> Plan --------------/
```

`core/interfaces.py` is the only shared surface (frozen — see below).
`member4_llm_rag/orchestrator.py` is the only file allowed to import across
all four member packages.

## What's implemented (Planning, #2)

`member2_planning/planning.py`:
- **`AStarPlanner`** — real A* over a tile-graph built from the level's
  ground array. Handles single-tile gaps via a jump edge, and wider gaps
  that only a moving platform can bridge via a `bridge_tiles` set (walkable
  at elevated cost for the macro route; exact timing is left to MCTS).
- **`MCTSPlanner`** — a lightweight rollout search (not a full UCT tree):
  samples short action-sequence rollouts against predicted hazard positions
  (moving platform phase, enemy patrol, extrapolated forward in time),
  scores by progress-toward-waypoint minus collision risk, and returns the
  best immediate action plus a risk estimate. Good enough to unblock the
  pipeline; upgrading to proper UCT selection is a clean drop-in later
  without touching the `Plan`/`PlanningModule` contract.
- **`HybridPlanningModule`** — combines both: A* gives the macro path,
  re-planned only when the agent drifts off the cached route; MCTS picks
  the next primitive `Action` each tick.

`demo/` — a small tile-based level (`level.py`) and a real (if simple)
`BaseEnv`-shaped environment (`demo_env.py`) with actual physics: walking,
single-tile jumps, a moving platform on a triangle-wave cycle, a patrolling
enemy, a collectible, a goal. `run_demo.py` drives this env with the real
planner (no RL yet — see Member 3's task below) and records a full tick-by-
tick trace to `frontend/trace.json`.

`frontend/` — `index.html` is a self-contained dashboard (canvas render +
playback controls + module status sidebar + live telemetry ticker) with the
trace baked in, so it opens directly in a browser with no server. Rebuild it
after changing the level or planner:

```bash
PYTHONPATH=. python3 frontend/build.py
```

This is scoped narrowly on purpose: `core/interfaces.py` and `core/env.py`
(the shared contract) are untouched, and the demo level/env live entirely
under `demo/` so they don't collide with anyone else's work or become the
assumed "real" level engine.

## Remaining work, by member

### Member 1 — Perception (`member1_perception/`)
Implement `CNNLSTMPerception.process()`:
1. CNN detector on frames from a level (start against `demo/demo_env.py`'s
   `render()` — currently a blank placeholder frame; swap in real sprites or
   feed it `DemoPlatformEnv`'s tile state rendered to pixels).
2. LSTM tracker for stable entity IDs + velocity across frames.
3. Trajectory forecasting (`predicted_trajectories`) for the moving platform
   and enemy — this can directly replace the `hazard_predictor` closure
   currently hardcoded inside `HybridPlanningModule.plan()`, which right now
   cheats by reading ground-truth `GameState.entities` instead of predicting
   from pixels.
See `member1_perception/README.md` for full milestones.

### Member 3 — RL Policy (`member3_rl/`)
Implement `PPOPolicy` / `DQNPolicy`:
1. Featurize `GameState` (+ optionally `Plan.waypoints`/`risk_score` from the
   now-working planner) into an obs vector.
2. Train PPO against `demo/demo_env.py` first (it's a real working env with
   actual reward signal already wired — `reward_total` in the trace). Right
   now `demo/run_demo.py`'s action source is the planner's `committed_action`
   directly; swap in `PPOPolicy.act(state, plan, perception)` once it exists,
   using the plan as a prior/auxiliary input rather than a hard override.
3. Randomize the level (vary `demo/level.py`'s gap positions, moving
   platform period/amplitude, enemy patrol range) and train across that
   distribution — the generalization requirement needs this baked in from
   the start, not added later.
4. DQN baseline for comparison.
See `member3_rl/README.md` for full milestones.

### Member 4 — LLM/RAG + Integration (`member4_llm_rag/`)
Implement `RAGMissionInterpreter.interpret()`:
1. Pick an LLM/SLM (local small model vs API) and a minimal KB format.
2. RAG retrieval + prompt template producing structured `MissionGuidance`.
3. `orchestrator.py` already runs the full Perception→Planning→Policy loop
   against stub/dummy modules (`tests/test_smoke.py` proves it); once
   Members 1 and 3 land real implementations, swap them in — no orchestrator
   changes should be needed if the interfaces are respected.
4. `HybridPlanningModule.plan()` accepts `guidance: MissionGuidance` already
   (it's in the signature) but doesn't use it yet — wire `constraints` and
   `reward_shaping_hints` into the MCTS scoring once this lands.
See `member4_llm_rag/README.md` for full milestones.

## Working independently

1. `core/interfaces.py` and `core/env.py` are frozen after the initial
   commit unless discussed as a group — changing a dataclass field breaks 3
   other people's code.
2. Each member works only inside their own `memberN_*/` folder + tests.
   `demo/` and `frontend/` are scoped to Member 2's work above; treat them
   as read-only reference unless you're extending the planner itself.
3. `git branch <name>/<feature>`, PR into `main`, at least one other member
   reviews before merge (rotate reviewers).
4. `tests/test_smoke.py` must keep passing — it's the integration contract,
   run it before every push: `PYTHONPATH=. pytest tests/`.
5. Weekly sync: each person demos against `MockPlatformEnv` (or `demo/` once
   your module plugs into it); swap in real level assets once the level
   engine (`assets/levels/`) exists — TBD who owns that, raise it in first
   sync if nobody's claimed it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/          # integration smoke test
PYTHONPATH=. python3 frontend/build.py   # regenerate the dashboard
```
