import os
import numpy as np
import torch

from core.env import MockPlatformEnv
from core.interfaces import Action
from member3_rl.policy import PPOPolicy
from member3_rl.trainPPO import evaluate  # reuse the exact same eval() used for the rigorous check


# ============================================================
# CONFIGURATION
# ============================================================

FINETUNE_EPISODES = 1000    # round 2: more episodes, same locked-difficulty regime
MAX_STEPS = 300
ROLLOUT_EPISODES = 10
EVAL_EVERY = 50
EVAL_EPISODES = 20

MODEL_DIR = "member3_rl/models"
LOAD_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_best.pth")  # round 1 result
FT_BEST_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_v2_best.pth")
FT_FINAL_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_finetuned_v2_final.pth")

# Continue from a seed range that does not overlap prior training (42..2542)
# or the rigorous eval (999000..999100).
BASE_SEED = 600000  # non-overlapping with round 1 (500000+) and rigorous eval (999000+)

env = MockPlatformEnv(seed=BASE_SEED)
eval_env = MockPlatformEnv(seed=BASE_SEED)


def finetune():
    os.makedirs(MODEL_DIR, exist_ok=True)

    policy = PPOPolicy()
    policy.agent.load(LOAD_PATH)
    print(f"Loaded checkpoint: {LOAD_PATH}")
    print(f"Device: {policy.agent.device}")

    # Establish the current rigorous baseline before touching anything,
    # so we only keep the fine-tuned model if it's an actual improvement.
    print("Running baseline rigorous eval (n=100) before fine-tuning...")
    baseline_reward, baseline_successes = evaluate(
        policy, start_seed=999000, episodes=100
    )
    print(f"Baseline: {baseline_successes}/100 successes, avg reward {baseline_reward:.3f}")
    print("=" * 65)

    rewards_history = []
    successful_episodes = 0
    best_success_rate = -1.0
    best_average_reward = -float("inf")

    episode = 0
    episodes_since_update = 0
    last_value = 0.0

    print("=" * 65)
    print("     PPO FINE-TUNE ROUND 2: LOCKED FULL DIFFICULTY (continued)")
    print("=" * 65)
    print(f"Episodes           : {FINETUNE_EPISODES}")
    print(f"Difficulty         : 1.0 (fixed, every episode)")
    print("=" * 65)

    while episode < FINETUNE_EPISODES:
        episode += 1
        episode_seed = BASE_SEED + episode
        env.rng = np.random.default_rng(episode_seed)

        env.set_difficulty(1.0)  # locked, no ramp

        state, frame = env.reset()
        episode_reward = 0.0
        success = False

        for step in range(MAX_STEPS):
            action, log_prob, value, observation = policy.act_for_rollout(
                state, None, None
            )
            next_state, next_frame, reward, done, info = env.step(action)

            policy.agent.remember(
                observation, action.value, log_prob, value, reward, done
            )

            episode_reward += reward

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
            eval_reward, eval_success = evaluate(
                policy, start_seed=800000 + episode, episodes=EVAL_EPISODES
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

    # Load whichever fine-tuned checkpoint scored best during fine-tuning
    # and compare it against the pre-finetune baseline on the SAME rigorous
    # protocol (n=100, seeds 999000+).
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
        print(">>> Keep the original best checkpoint; investigate reward shaping instead.")

    print("=" * 65)


if __name__ == "__main__":
    finetune()
