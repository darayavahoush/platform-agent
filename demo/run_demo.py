"""Runs the REAL planning module (member2_planning) against the demo tile
level, driven by the planner's committed_action each tick (no RL policy yet
— member3's PPO/DQN replaces this loop's action source later). Records a
full trace to frontend/trace.json for the dashboard to play back.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.demo_env import DemoPlatformEnv
from demo.level import (GROUND, HAZARD_TILES, BRIDGE_TILES, TILE_W, N_TILES,
                         COLLECTIBLE_TILE, GOAL_TILE, MOVING_PLATFORM, ENEMY)
from member1_perception.perception import CNNLSTMPerception
from member2_planning.planning import HybridPlanningModule


def main():
    env = DemoPlatformEnv()
    planner = HybridPlanningModule()
    perception = CNNLSTMPerception(history_len=8, horizon=5)
    planner.set_level(GROUND, HAZARD_TILES, BRIDGE_TILES)

    state, frame = env.reset()
    history = []
    trace = {
        "level": {
            "tile_w": TILE_W, "n_tiles": N_TILES, "ground": GROUND,
            "moving_platform": MOVING_PLATFORM, "enemy": ENEMY,
            "collectible_tile": COLLECTIBLE_TILE, "goal_tile": GOAL_TILE,
        },
        "ticks": [],
    }

    for _ in range(400):
        history.append(frame)
        if len(history) > perception.history_len:
            history = history[-perception.history_len:]

        perception_output = perception.process(frame, history)

        def hazard_predictor(t_offset: int) -> set[int]:
            hazards: set[int] = set()
            for entity in perception_output.entities:
                if entity.kind not in {"moving_platform", "enemy"}:
                    continue
                track_id = entity.extra.get("id")
                if track_id is None:
                    continue
                trajectory = perception_output.predicted_trajectories.get(track_id, [])
                if not trajectory:
                    continue
                if t_offset >= len(trajectory):
                    continue
                pred_x, _ = trajectory[t_offset]
                hazards.add(round(pred_x / TILE_W))
            return hazards

        planner.set_hazard_predictor(hazard_predictor)
        plan = planner.plan(state)
        action = plan.committed_action
        state, frame, reward, done, info = env.step(action)

        trace["ticks"].append({
            "t": state.tick,
            "player_tile": round(state.player.x / TILE_W),
            "action": action.name,
            "risk": round(plan.risk_score, 3),
            "waypoints": [round(w[0] / TILE_W) for w in plan.waypoints],
            "reward": round(reward, 3),
            "reward_total": round(state.info["reward_total"], 2),
            "collected": info["collected"],
            "hits": info["hits"],
        })
        if done:
            break

    out_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "trace.json")
    with open(out_path, "w") as f:
        json.dump(trace, f)

    final = trace["ticks"][-1]
    print(f"Trace written: {len(trace['ticks'])} ticks, "
          f"reached goal={round(final['player_tile']) >= GOAL_TILE}, "
          f"collected={final['collected']}, hits={final['hits']}, "
          f"reward_total={final['reward_total']}")


if __name__ == "__main__":
    main()
