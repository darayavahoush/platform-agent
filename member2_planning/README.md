# Member 2 — Planning (A* + MCTS)

**Owns:** `PlanningModule` in `core/interfaces.py`. Output type: `Plan`.

## Scope
- A*: discretize the level (from `GameState.entities` — platforms/gaps) into a graph, produce a coarse waypoint path to `GameState.goal`. Re-plan when the graph changes materially (moving platform phase shift, new hazard).
- MCTS: given the next waypoint and predicted trajectories of moving obstacles, search over `Action` sequences for the next committed action + `risk_score`. This is what actually gets fed to Member 3's policy as guidance/prior.
- Consume `MissionGuidance.constraints` / `reward_shaping_hints` if present to bias search (e.g. avoid regions, prioritize collectibles).

## Milestones
1. Graph construction from `GameState` on `MockPlatformEnv` + A* shortest path.
2. Replanning trigger logic (state delta threshold / fixed interval).
3. MCTS rollout policy using a simple heuristic (or later, Member 3's policy net as rollout policy) for tactical action choice.
4. Integrate predicted trajectories (stub these locally with straight-line extrapolation until Member 1's output is available — don't block on them).

## Test independently
```
from core.env import MockPlatformEnv
from member2_planning.planning import HybridPlanningModule
env = MockPlatformEnv(); state, _ = env.reset()
plan = HybridPlanningModule().plan(state)
```
