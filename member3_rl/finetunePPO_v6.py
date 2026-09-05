import os
import math
import numpy as np
import torch

from core.env import MockPlatformEnv
from core.interfaces import Action
from member3_rl.policy import PPOPolicy
from member3_rl.trainPPO import evaluate  # reuse the exact same eval() used for the rigorous check


# ============================================================
# CONFIGURATION
# ============================================================

FINETUNE_EPISODES = 1000
MAX_STEPS = 300
ROLLOUT_EPISODES = 10
EVAL_EVERY = 50
EVAL_EPISODES = 20

MODEL_DIR = "member3_rl/models"
# NOTE: loading v3_best, not v4 — round 4 showed no improvement over v3,
# so v3 is still the last genuinely better checkpoint.
LOAD_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_v3_best.pth")
FT_BEST_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_v6_best.pth")
FT_FINAL_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_v6_final.pth")

# Fresh, clearly non-overlapping seed blocks (previous rounds used
# 500000-1000000 for training and 700000-1000000+ for eval offsets).
BASE_SEED = 2500000
EVAL_OFFSET = 2600000

# ------------------------------------------------------------
# Enemy-avoidance shaping (potential-based, training-only)
# ------------------------------------------------------------
# phi(state) in [0, 1]: 1.0 = no enemy nearby / far away, 0.0 = right on
# top of an enemy. Potential-based shaping (gamma*phi(next) - phi(state))
# does not change the optimal policy (Ng et al., 1999) -- it only makes
# the "don't stand next to an enemy" signal dense instead of sparse.
SAFE_DIST = 100.0
SHAPING_WEIGHT = 0.4


def _nearest_enemy_dist(state):
    enemies = [e for e in state.entities if e.kind == "enemy"]
    if not enemies:
        return SAFE_DIST
    player = state.player
    dists = [
        math.hypot(e.x - player.x, e.y - player.y)
        for e in enemies
    ]
    return min(dists)


def _phi(state):
    d = min(_nearest_enemy_dist(state), SAFE_DIST)
    return d / SAFE_DIST


env = MockPlatformEnv(seed=BASE_SEED)
eval_env = MockPlatformEnv(seed=BASE_SEED)


def finetune():
    os.makedirs(MODEL_DIR, exist_ok=True)

    policy = PPOPolicy()
    policy.agent.load(LOAD_PATH)
    print(f"Loaded checkpoint: {LOAD_PATH}")
    print(f"Device: {policy.agent.device}")
    print(f"Enemy-avoidance shaping: weight={SHAPING_WEIGHT}, safe_dist={SAFE_DIST}")

    print("Running baseline rigorous eval (n=100) before fine-tuning...")
    baseline_reward, baseline_successes = evaluate(
        policy, start_seed=999000, episodes=100
    )
    print(f"Baseline: {baseline_successes}/100 successes, avg reward {baseline_reward:.3f}")
    print("=" * 65)

    rewards_history = []       # raw env reward (for reporting, comparable to prior rounds)
    successful_episodes = 0
    best_success_rate = -1.0
    best_average_reward = -float("inf")

    episode = 0
    episodes_since_update = 0
    last_value = 0.0

    print("=" * 65)
    print("  PPO FINE-TUNE ROUND 6: ENEMY-AVOIDANCE SHAPING v2 (rescaled, from v3_best)")
    print("=" * 65)
    print(f"Episodes           : {FINETUNE_EPISODES}")
    print(f"Difficulty         : 1.0 (fixed, every episode)")
    print("=" * 65)

    while episode < FINETUNE_EPISODES:
        episode += 1
        episode_seed = BASE_SEED + episode
        env.rng = np.random.default_rng(episode_seed)
        env.set_difficulty(1.0)

        state, frame = env.reset()
        episode_reward = 0.0       # raw, for logging/comparison
        success = False

        for step in range(MAX_STEPS):
            action, log_prob, value, observation = policy.act_for_rollout(
                state, None, None
            )
            phi_before = _phi(state)

            next_state, next_frame, reward, done, info = env.step(action)

            phi_after = _phi(next_state)
            shaping = SHAPING_WEIGHT * (
                policy.agent.gamma * phi_after - phi_before
            )
            shaping = float(np.clip(shaping, -0.1, 0.1))  # never louder than landing/progress reward
            shaped_reward = reward + shaping

            policy.agent.remember(
                observation, action.value, log_prob, value, shaped_reward, done
            )

            episode_reward += reward  # raw reward tracked for reporting

            if info.get("reached_goal", False):
                success = True

            state = next_state
            frame = next_frame

            if done:
                last_value = 0.0
                break
        else:
            next_obs = policy.featurizer.transform(state, None, None)
            with torch.no_grad():
                next_obs_t = torch.tensor(
                    next_obs, dtype=torch.float32,
                    device=policy.agent.device
                ).unsqueeze(0)
                _, last_value_t = policy.agent.network.forward(next_obs_t)
                last_value = float(last_value_t.item())

        rewards_history.append(episode_reward)
        if success:
            successful_episodes += 1
        episodes_since_update += 1

        if episode <= 10 or episode % 10 == 0:
            recent_avg = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode:4d}/{FINETUNE_EPISODES} | "
                f"Steps: {step + 1:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg10: {recent_avg:7.2f} | "
                f"Success: {'YES' if success else 'NO'}"
            )

        if episodes_since_update >= ROLLOUT_EPISODES:
            loss = policy.agent.learn(last_value)
            episodes_since_update = 0
            if loss is not None:
                print(f"    [ppo update] avg_loss={loss:.4f}")

        if episode % EVAL_EVERY == 0:
            # Eval uses the plain evaluate() — raw reward only, no shaping —
            # so these numbers are directly comparable to every prior round.
            eval_reward, eval_success = evaluate(
                policy, start_seed=EVAL_OFFSET + episode, episodes=EVAL_EPISODES
            )
            eval_success_rate = (eval_success / EVAL_EPISODES) * 100.0

            print()
            print("-" * 65)
            print(f"EVALUATION @ Fine-tune Episode {episode}")
            print(f"Average reward : {eval_reward:.3f}")
            print(
                f"Success rate   : {eval_success}/{EVAL_EPISODES} "
                f"({eval_success_rate:.1f}%)"
            )
            print("-" * 65)

            if (
                eval_success_rate > best_success_rate
                or (
                    eval_success_rate == best_success_rate
                    and eval_reward > best_average_reward
                )
            ):
                best_success_rate = eval_success_rate
                best_average_reward = eval_reward
                policy.agent.save(FT_BEST_PATH)
                print(">>> NEW FINE-TUNED BEST MODEL SAVED")
                print()

    policy.agent.save(FT_FINAL_PATH)

    print()
    print("=" * 65)
    print("            FINE-TUNING COMPLETE — RUNNING FINAL RIGOROUS EVAL")
    print("=" * 65)

    final_policy = PPOPolicy()
    load_path = FT_BEST_PATH if os.path.exists(FT_BEST_PATH) else FT_FINAL_PATH
    final_policy.agent.load(load_path)

    ft_reward, ft_successes = evaluate(final_policy, start_seed=999000, episodes=100)

    print()
    print(f"BEFORE fine-tune : {baseline_successes}/100 (avg reward {baseline_reward:.3f})")
    print(f"AFTER  fine-tune : {ft_successes}/100 (avg reward {ft_reward:.3f})")
    print()

    if ft_successes > baseline_successes:
        print(">>> Fine-tuning IMPROVED rigorous success rate. Consider promoting")
        print(f">>> '{load_path}' to replace '{LOAD_PATH}'.")
    else:
        print(">>> Fine-tuning did NOT improve rigorous success rate.")
        print(">>> Enemy shaping weight/safe_dist may need tuning, or this needs")
        print(">>> a different intervention (e.g. per-enemy detection radius).")

    print("=" * 65)


if __name__ == "__main__":
    finetune()
