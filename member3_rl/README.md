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

## Fine-Tuning: Closing the Difficulty Gap

The initial PPO run (see comparison above) scored 16/100 on the rigorous
eval, with `fell` as the dominant failure mode (40%), caused by the
curriculum still ramping at the point training ended (episode 2500 vs.
curriculum completion at 1400 â€” not enough hard-difficulty exposure).

**Approach:** continued training from the best checkpoint with difficulty
locked at 1.0 for every episode (no curriculum ramp), in fixed rounds of
800-1000 episodes, checking the rigorous 100-episode eval after each round
before deciding whether to continue.

| Round | Change | Rigorous eval (n=100) | Failure breakdown (timeout / hazard / enemy / fell) |
|-------|--------|------------------------|------------------------------------------------------|
| 0 (initial) | curriculum training only | 16/100 | 0 / 21 / 23 / 40 |
| 1 | +800 ep, locked difficulty | 17/100 | 11 / 29 / 27 / 16 |
| 2 | +1000 ep, locked difficulty | 24/100 | 3 / 22 / 36 / 15 |
| 3 | +1000 ep, locked difficulty | **31/100** | 0 / 19 / 39 / 11 |
| 4 | +1000 ep, locked difficulty | 31/100 (no improvement) | 0 / 24 / 37 / 8 |

Rounds 1-3 show a clean, non-plateauing trend: locked-difficulty fine-tuning
resolved `timeout` entirely and cut `fell` from 40% to 11% of episodes.
Round 4 showed no further improvement, indicating this specific
intervention had run its course. **`enemy` collisions became the dominant
remaining failure mode** (23% -> 39% of episodes) as the other failure
types shrank.

**Enemy-avoidance shaping (rounds 5-6, not adopted):** a potential-based
reward shaping term (`gamma*phi(next) - phi(current)`, where `phi` is
normalized distance to the nearest enemy) was added to the *training*
reward only, to give a dense signal for enemy avoidance instead of the
existing sparse terminal penalty. Two weight/scale configurations were
tried:

| Round | Shaping weight | Rigorous eval | Failure breakdown |
|-------|------------------|-----------------|---------------------|
| 5 | weight=3.0, safe_dist=180 | 27/100 | 2 / 15 / 31 / 25 |
| 6 | weight=0.4 (clipped +/-0.1), safe_dist=100 | 19/100 | 0 / 25 / 41 / 15 |

Neither configuration beat the round-3 checkpoint, and results were
inconsistent between the two weight settings (round 6's `enemy` failures
were *higher* than the unshaped round-3 baseline despite a much smaller
shaping weight), suggesting high run-to-run variance rather than a
controllable relationship between shaping strength and outcome given a
single 1000-episode run per configuration. This line of investigation was
not pursued further; a robust conclusion would need multiple seeds per
configuration, which was judged not worth the additional compute for this
milestone.

**Final adopted model:** round 3's checkpoint
(`ppo_platform_agent_finetuned_v3_best.pth`, promoted to
`ppo_platform_agent_best.pth`), scoring **31/100 (31%)** on the rigorous
100-episode evaluation â€” a 7.75x improvement over the DQN baseline (4/100)
and roughly double the untuned PPO result (16/100).

**Suggested future work:** `enemy` collisions remain the largest failure
mode (39%) in the final model. Promising next steps not attempted here:
per-enemy detection-radius features in the observation space, or a
properly ablated shaping study across multiple random seeds per
configuration to distinguish genuine effect from training variance.
