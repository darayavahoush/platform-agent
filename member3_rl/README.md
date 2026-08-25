# Member 3 — Reinforcement Learning (PPO/DQN)

**Owns:** `PolicyModule` in `core/interfaces.py`.

## Scope
- Build a Gym-style training loop around `core.env.BaseEnv` (start with `MockPlatformEnv`).
- Featurize `GameState` (+ optionally `PerceptionOutput.embedding`, `Plan.waypoints`/`risk_score`) into a fixed obs vector.
- Train PPO as primary policy; DQN as a baseline for comparison.
- Reward: base env reward + shaping terms from `MissionGuidance.reward_shaping_hints` (Member 4) — keep this pluggable, don't hardcode.
- Generalization: train over a **distribution** of procedurally-varied mock levels (randomize gaps/platform phase/enemy patrol), not one fixed layout — this is the generalization-to-unseen-levels requirement, so it has to be baked into the training setup from the start, not bolted on later.

## Milestones
1. Featurizer: `GameState -> np.ndarray`.
2. PPO loop on `MockPlatformEnv` (single fixed layout) — sanity check learning at all.
3. Level randomization wrapper -> re-train, hold out a test set of seeds, report generalization gap.
4. DQN baseline for comparison; log both to `tests/` metrics.
5. Wire in `Plan` (as auxiliary input or action mask) and `PerceptionOutput.embedding` once Members 1/2 expose stable outputs — build against interface stubs until then, don't block.

## Test independently
```
from core.env import MockPlatformEnv
from member3_rl.policy import PPOPolicy
env = MockPlatformEnv(); state, _ = env.reset()
policy = PPOPolicy()
action = policy.act(state)
```
