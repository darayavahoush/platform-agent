# Member 1 — Perception (CNN + LSTM)

**Owns:** `PerceptionModule` in `core/interfaces.py`. Output type: `PerceptionOutput`.

## Scope
- CNN: per-frame detection/segmentation of player, platforms, enemies, collectibles, hazards from `Frame.rgb`.
- LSTM: consume a rolling window of CNN features/detections to (a) track entity identity across frames, (b) predict short-horizon trajectories for moving platforms and enemies (`predicted_trajectories`).
- Expose an optional latent `embedding` other modules (esp. RL) can consume instead of raw state.

## Milestones
1. CNN detector on synthetic/rendered frames from `MockPlatformEnv` (bounding boxes or heatmaps).
2. LSTM tracker: stable entity IDs across frames + velocity estimate.
3. Trajectory forecasting: predict next H positions for moving platforms/enemies, report `confidence`.
4. Swap in real level renderer once available; no interface changes needed if `Frame` contract is respected.

## Test independently
```
from core.env import MockPlatformEnv
from member1_perception.perception import CNNLSTMPerception
env = MockPlatformEnv(); state, frame = env.reset()
p = CNNLSTMPerception()
out = p.process(frame, history=[frame])
```
Don't block on real assets — build/test against `MockPlatformEnv` frames first.
