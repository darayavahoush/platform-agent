# Member 4 — LLM/RAG Strategic Layer + Integration

**Owns:** `MissionModule` in `core/interfaces.py`, and `orchestrator.py` (the ONLY file allowed to import all 4 member packages).

## Scope
- RAG: build a small knowledge base (level docs, strategy heuristics, past-run postmortems) + retriever.
- LLM/SLM: given `GameState` + retrieved context + a natural-language mission (e.g. "collect all gems, avoid unnecessary combat, reach the goal fast"), produce `MissionGuidance` (subgoal, reward shaping hints, constraints).
- Keep LLM calls infrequent (`guidance_refresh_every`) — it's strategic, not per-tick.
- Own `orchestrator.py`: the control loop wiring Perception -> Planning -> Policy -> Env each tick, refreshing guidance periodically. This is integration work and should track the other 3 members' interfaces as they stabilize.

## Milestones
1. Pick LLM/SLM (local small model vs API) and a minimal KB format.
2. RAG retrieval + prompt template producing structured `MissionGuidance` (JSON-constrained).
3. `orchestrator.py` end-to-end run against all 4 stub modules on `MockPlatformEnv` (proves the pipeline, even with `NotImplementedError` stubs replaced by trivial dummy logic for the smoke test).
4. Swap dummy modules for real ones as teammates land them — interfaces mean no orchestrator changes needed.

## Test independently
Run against dummy/trivial implementations of the other 3 interfaces first (don't wait on teammates) — e.g. a `PlanningModule` that returns a straight line, a `PolicyModule` that always returns `Action.RIGHT`. Confirms your orchestration logic and MissionGuidance plumbing work before real modules exist.
