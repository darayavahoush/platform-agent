"""Runs the REAL trained Member 3 policy (PPO, greedy/deterministic) against
its native training environment, core.env.MockPlatformEnv — a continuous-
physics 900x500 world with real gravity and jump arcs. This is a DIFFERENT
world than demo/demo_env.py (the flat tile grid used by run_demo.py for the
Planning-only replay): the two are not observation/action compatible, so the
trained RL checkpoints cannot be dropped into that tile loop as-is.

Records a full trace to frontend/public/rl_trace.json for the dashboard's
"RL Agent" view to play back. Static level geometry (platforms/hazards/goal)
is recorded once; only entities that move each tick (player, moving
platforms, enemies) are recorded per-tick.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import MockPlatformEnv
from member3_rl.policy import PPOPolicy

MODEL_PATH = os.path.join("member3_rl", "models", "ppo_platform_agent_best.pth")
MAX_STEPS = 300

# Seeds are unseen relative to training (matches the 999000-range rigorous
# eval seeds used for the PPO-vs-DQN comparison in the README). Not every
# seed is a success — the agent's real eval-time success rate is ~31-36%,
# not 100% — so we scan a small unseen-seed range and pick a genuine success
# to showcase, the same way a human would pick a representative run rather
# than a cherry-picked outlier.
CANDIDATE_SEEDS = range(999000, 999030)


def entity_dict(e):
    return {
        "x": round(float(e.x), 1),
        "y": round(float(e.y), 1),
        "width": e.extra.get("width", 0),
        "height": e.extra.get("height", 0),
    }


def run_episode(policy, seed):
    env = MockPlatformEnv(seed=seed)
    state, frame = env.reset()

    trace = {
        "world": {"width": env.WIDTH, "height": env.HEIGHT},
        "player_size": {"w": env.PLAYER_W, "h": env.PLAYER_H},
        "seed": seed,
        "static": {
            "platforms": [entity_dict(p) for p in env.platforms],
            "hazards": [entity_dict(h) for h in env.hazards],
            "goal": [round(env.goal[0], 1), round(env.goal[1], 1)],
        },
        "ticks": [],
    }

    reward_total = 0.0
    outcome = {"reached_goal": False, "steps": 0}

    for _ in range(MAX_STEPS):
        action = policy.act(state)
        next_state, frame, reward, done, info = env.step(action)
        reward_total += float(reward)

        trace["ticks"].append({
            "t": next_state.tick,
            "player": {"x": round(env.player.x, 1), "y": round(env.player.y, 1)},
            "moving_platforms": [entity_dict(m) for m in env.moving_platforms],
            "enemies": [entity_dict(e) for e in env.enemies],
            "action": action.name,
            "reward": round(reward, 3),
            "reward_total": round(reward_total, 2),
        })

        state = next_state
        if done:
            outcome = {
                "reached_goal": bool(info.get("reached_goal", False)),
                "steps": next_state.tick,
                "reason": info.get("reason", "goal" if info.get("reached_goal") else "timeout"),
            }
            break
    else:
        outcome = {"reached_goal": False, "steps": MAX_STEPS, "reason": "timeout"}

    trace["outcome"] = {**outcome, "reward_total": round(reward_total, 2)}
    return trace


def main():
    policy = PPOPolicy()
    policy.agent.load(MODEL_PATH)

    chosen = None
    for seed in CANDIDATE_SEEDS:
        trace = run_episode(policy, seed)
        if trace["outcome"]["reached_goal"]:
            chosen = trace
            break

    if chosen is None:
        # Honest fallback: no success in the scanned range, use the longest
        # attempt rather than silently faking a result.
        chosen = max(
            (run_episode(policy, s) for s in CANDIDATE_SEEDS),
            key=lambda t: t["outcome"]["steps"],
        )

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "public", "rl_trace.json"
    )
    with open(out_path, "w") as f:
        json.dump(chosen, f)

    o = chosen["outcome"]
    print(
        f"RL trace written: seed={chosen['seed']} ticks={len(chosen['ticks'])} "
        f"reached_goal={o['reached_goal']} reward_total={o['reward_total']}"
    )


if __name__ == "__main__":
    main()
