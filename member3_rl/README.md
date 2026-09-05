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

## PPO vs. DQN — Final Comparison

Both algorithms were evaluated with the same rigorous protocol: deterministic
(greedy, no exploration) policy, full difficulty (1.0), 100 episodes,
disjoint seed range from training.

| Metric                        | DQN (baseline) | PPO (primary) |
|--------------------------------|----------------|----------------|
| Training-time success rate     | 15.12%         | 24.56%         |
| Best small-sample eval (n=20)  | 50.0%          | 30.0%          |
| Rigorous eval (n=100)          | 4/100 (4.0%)   | 16/100 (16.0%) |
| Rigorous eval avg reward       | —              | 15.348         |

**Headline result:** PPO outperforms DQN by 4x on the rigorous 100-episode
evaluation (16% vs. 4% success rate), which is the number that should be
cited as the primary result, not the smaller-sample checkpoint evals.

**Why the small-sample numbers are misleading:** both algorithms' best
checkpoints looked stronger under a 20-episode eval than they held up under
a 100-episode eval (DQN: 50% -> 4%, PPO: 30% -> 16%). This is expected
variance at n=20 and is the reason a larger, fixed-seed evaluation was run
before reporting final numbers.

**Failure mode breakdown (PPO, n=100, full difficulty):**
{'timeout': 0, 'hazard': 21, 'enemy': 23, 'fell': 40}

No timeouts occurred, meaning the agent reliably reaches hazards/enemies
rather than stalling. "Fell" remains the dominant failure mode (40%),
consistent with earlier training-time evals, indicating the hardest
gap-crossing / vertical-jump sections of the curriculum are still not
fully solved even after curriculum completion (episode 1400) and 1100
additional training episodes.

**Training stability:** PPO loss peaked at ~2.9 around episode 400, then
settled into a 0.4-1.8 range for the remainder of training with no
divergence — a healthier training curve than DQN's early-training
instability.

**Artifacts:**
- Best model: `member3_rl/models/ppo_platform_agent_best.pth`
- Final model: `member3_rl/models/ppo_platform_agent_final.pth`
- Training script: `member3_rl/trainPPO.py`
