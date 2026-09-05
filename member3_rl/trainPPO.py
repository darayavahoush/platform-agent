import os
import random
import numpy as np
import torch

from core.env import MockPlatformEnv
from core.interfaces import Action
from member3_rl.policy import PPOPolicy


# ============================================================
# CONFIGURATION
# ============================================================

CURRICULUM_EPISODES = 1400
NUM_EPISODES = 2500
MAX_STEPS = 300

ROLLOUT_EPISODES = 10   # collect this many episodes before each PPO update

EVAL_EVERY = 50
EVAL_EPISODES = 20

MODEL_DIR = "member3_rl/models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_platform_agent_final.pth")

BASE_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)


# ============================================================
# ENVIRONMENT
# ============================================================

env = MockPlatformEnv(seed=BASE_SEED)
eval_env = MockPlatformEnv(seed=BASE_SEED)

policy = PPOPolicy()


# ============================================================
# TRAINING
# ============================================================

def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    rewards_history = []
    successful_episodes = 0
    best_average_reward = -float("inf")
    best_success_rate = 0.0

    print("=" * 65)
    print("           PPO PLATFORM NAVIGATION TRAINING")
    print("=" * 65)
    print(f"Episodes           : {NUM_EPISODES}")
    print(f"Max steps          : {MAX_STEPS}")
    print(f"Rollout size       : {ROLLOUT_EPISODES} episodes/update")
    print(f"Evaluation         : Every {EVAL_EVERY} episodes")
    print(f"Device             : {policy.agent.device}")
    print("=" * 65)

    episode = 0
    episodes_since_update = 0
    last_value = 0.0

    while episode < NUM_EPISODES:

        episode += 1
        episode_seed = BASE_SEED + episode
        env.rng = np.random.default_rng(episode_seed)

        difficulty = min(1.0, episode / CURRICULUM_EPISODES)
        env.set_difficulty(difficulty)

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
            # Episode hit MAX_STEPS without env-side done (shouldn't happen
            # since env's own MAX_STEPS matches, but bootstrap safely if so).
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

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if episode <= 10 or episode % 10 == 0:
            recent_avg = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode:4d}/{NUM_EPISODES} | "
                f"Steps: {step + 1:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg10: {recent_avg:7.2f} | "
                f"Success: {'YES' if success else 'NO'}"
            )

        # ----------------------------------------------------
        # PPO update after collecting enough episodes
        # ----------------------------------------------------

        if episodes_since_update >= ROLLOUT_EPISODES:
            loss = policy.agent.learn(last_value)
            episodes_since_update = 0
            if loss is not None:
                print(f"    [ppo update] avg_loss={loss:.4f}")

        # ----------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------

        if episode % EVAL_EVERY == 0:

            eval_reward, eval_success = evaluate(
                policy, start_seed=10000 + episode, episodes=EVAL_EPISODES
            )
            eval_success_rate = (eval_success / EVAL_EPISODES) * 100.0

            print()
            print("-" * 65)
            print(f"EVALUATION @ Episode {episode}")
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
                policy.agent.save(BEST_MODEL_PATH)
                print(">>> NEW BEST MODEL SAVED")
                print()

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    policy.agent.save(FINAL_MODEL_PATH)

    overall_success_rate = (successful_episodes / NUM_EPISODES) * 100.0
    average_reward = np.mean(rewards_history)
    last_10_average = (
        np.mean(rewards_history[-10:])
        if len(rewards_history) >= 10
        else average_reward
    )

    print()
    print("=" * 65)
    print("                 TRAINING COMPLETE")
    print("=" * 65)
    print(f"Successful episodes : {successful_episodes}/{NUM_EPISODES}")
    print(f"Training success    : {overall_success_rate:.2f}%")
    print(f"Average reward      : {average_reward:.3f}")
    print(f"Last 10 avg reward  : {last_10_average:.3f}")
    print(f"Best eval success   : {best_success_rate:.1f}%")
    print(f"Best eval reward    : {best_average_reward:.3f}")
    print()
    print("Best model:")
    print(BEST_MODEL_PATH)
    print()
    print("Final model:")
    print(FINAL_MODEL_PATH)
    print("=" * 65)


# ============================================================
# EVALUATION (deterministic / greedy, no exploration)
# ============================================================

def evaluate(policy, start_seed=10000, episodes=20):

    total_reward = 0.0
    successful = 0
    fail_reasons = {"timeout": 0, "hazard": 0, "enemy": 0, "fell": 0, "other": 0}

    for i in range(episodes):
        seed = start_seed + i
        eval_env.rng = np.random.default_rng(seed)
        eval_env.set_difficulty(1.0)  # always eval at full difficulty

        state, frame = eval_env.reset()
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            action = policy.act(state, None, None)
            next_state, next_frame, reward, done, info = eval_env.step(action)

            episode_reward += reward
            state = next_state
            frame = next_frame

            if info.get("reached_goal", False):
                successful += 1

            if done:
                if not info.get("reached_goal", False):
                    reason = info.get(
                        "reason", "timeout" if info.get("timeout") else "other"
                    )
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                break

        total_reward += episode_reward

    average_reward = total_reward / episodes
    print(f"  Failure breakdown: {fail_reasons}")

    return average_reward, successful


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
