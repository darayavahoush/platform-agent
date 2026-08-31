# Member 1 — Perception (CNN + LSTM)

**Owns:** `PerceptionModule` in `core/interfaces.py`. Output type: `PerceptionOutput`.

## Member 1 responsibility
- CNN visual perception: extract image features from each frame and infer the relevant scene elements from `Frame.rgb`.
- LSTM temporal modeling: consume a rolling history of visual features to maintain temporal context beyond a single image.
- Entity tracking: maintain stable entity IDs across frames so the same object is consistently represented over time.
- Velocity estimation: estimate per-entity motion from frame-to-frame position deltas.
- Trajectory forecasting: forecast future positions for moving-platform and enemy entities over a configurable horizon.

## Current implementation
`Frame.rgb` is the direct visual input to Member 1. Each frame is a synthetic RGB image array, and a small PyTorch CNN extracts compact feature vectors from that image. The network is intentionally lightweight and acts as a visual feature extractor rather than a trained object-detection model.

The current demo detector is synthetic and color-based. Because the demo renderer uses simple colored shapes, the perception module identifies entities by thresholding pixel colors and extracting bounding boxes from masks. This is not a general-purpose detector; it is a pragmatic solution for the demo scene and can be replaced later with a stronger visual detector without changing the public `Frame` contract.

The CNN output is combined with an LSTM that processes a rolling sequence of recent CNN features. This temporal model is used to maintain a coherent latent trajectory context and to estimate short-horizon motion. Stable entity IDs are maintained across frames by matching detections to prior tracked objects using position and size consistency. Velocity is estimated from the difference between current and previous positions divided by the frame delta, and moving-platform and enemy trajectories are forecast over a configurable horizon via a lightweight forecasting head.

The final result from Member 1 is `PerceptionOutput`, which provides structured detections, player pose, forecasted paths, confidence, and a latent embedding for downstream modules.

## Output contract
`PerceptionOutput` is the shared output contract from Member 1. It includes:

- `entities`: the list of tracked/detected `Entity` objects, each carrying kind, position, velocity, and metadata such as ID, bounding-box size, and confidence.
- `player_pose`: the current `Entity` representing the player, used by downstream planning and policy logic.
- `predicted_trajectories`: a mapping from entity ID to a future list of `(x, y)` coordinates over the configured forecast horizon.
- `confidence`: a scalar quality estimate derived from the current detection set.
- `embedding`: the temporal embedding produced by the LSTM/CNN pipeline, currently a NumPy vector with shape `(64,)` in the demo implementation.

## Planner integration
`predicted_trajectories` are converted from image/world-space `x/y` coordinates into planner tile indices before tactical hazard reasoning. In the demo, `demo/run_demo.py` constructs the hazard predictor directly from the perception output so planner risk estimation can account for the moving entities that perception has tracked and forecast.

`HybridPlanningModule.set_hazard_predictor()` allows the perception-derived predictor to be supplied instead of the default internal logic. This is the integration point that lets external forecasting information influence planning. At the same time, `member2_planning/planning.py` retains its existing state-based predictor as a backward-compatible fallback when no external predictor is supplied.

## Testing
The implementation was validated over 50 frames in the demo flow and the checks covered:
- stable IDs were maintained across frames
- velocities were produced for tracked entities
- embedding shape was `(64,)`
- trajectory length matched the configured horizon
- there is no `GameState` or `state.entities` access in `perception.py`
- there are no cross-member imports in Member 1 code

This is a lightweight validation of the current demo pipeline rather than a claim of production readiness.

## Demo usage
Run the demo from the project root:

```bash
python demo/run_demo.py
```

To run the frontend visualization:

```bash
cd frontend
npm install
npm run dev
```

## Current limitations
- The detector is currently synthetic and color-based because the demo renderer uses simple colored shapes rather than full art assets.
- The CNN is not a trained general object detector; it is a feature extractor for the demo pipeline.
- Forecasting is prototype-level and not trained on a large trajectory dataset.
- Tracking is not designed for severe occlusion or robust re-identification after long disappearances.
- Real sprites or richer rendering can be substituted later without changing the `Frame` contract.

## Test independently
The original lightweight smoke test remains valid for local testing:

```python
from core.env import MockPlatformEnv
from member1_perception.perception import CNNLSTMPerception

env = MockPlatformEnv()
state, frame = env.reset()
p = CNNLSTMPerception()
out = p.process(frame, history=[frame])
```

This keeps Member 1 testable without waiting on final assets or a fully trained detector pipeline.
