import os
import random
import numpy as np
import torch

from core.env import MockPlatformEnv
from core.interfaces import Action
from member3_rl.policy import DQNPolicy


# ============================================================
# CONFIGURATION
# ============================================================

CURRICULUM_EPISODES = 1400
NUM_EPISODES = 500
MAX_STEPS = 300

EVAL_EVERY = 50
EVAL_EPISODES = 20

MODEL_DIR = "member3_rl/models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_platform_agent_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_platform_agent_final.pth")

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

policy = DQNPolicy()


# ============================================================
# TRAINING
# ============================================================

def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    rewards_history = []
    losses_history = []

    successful_episodes = 0
    best_average_reward = -float("inf")
    best_success_rate = 0.0

    print("=" * 65)
    print("           DQN PLATFORM NAVIGATION TRAINING")
    print("=" * 65)
    print(f"Episodes       : {NUM_EPISODES}")
    print(f"Max steps      : {MAX_STEPS}")
    print(f"Evaluation     : Every {EVAL_EVERY} episodes")
    print(f"Device         : {policy.agent.device}")
    print("=" * 65)

    for episode in range(1, NUM_EPISODES + 1):

        # Different seed for every training episode
        episode_seed = BASE_SEED + episode

        env.rng = np.random.default_rng(episode_seed)

        difficulty = min(1.0, episode / CURRICULUM_EPISODES)
        env.set_difficulty(difficulty)

        state, frame = env.reset()

        episode_reward = 0.0
        episode_losses = []

        success = False

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # IMPORTANT:
            # Create observation BEFORE env.step()
            # ------------------------------------------------

            observation = policy.featurizer.transform(
                state,
                None,
                None
            )

            # Select action using epsilon-greedy policy
            action_index = policy.agent.select_action(
                observation,
                training=True
            )

            action = Action(action_index)

            # Environment transition
            next_state, next_frame, reward, done, info = env.step(action)

            # Create next observation AFTER env.step()
            next_observation = policy.featurizer.transform(
                next_state,
                None,
                None
            )

            # Store transition
            policy.agent.remember(
                observation,
                action_index,
                reward,
                next_observation,
                done
            )

            # Train DQN
            loss = policy.agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward

            # Check success
            if info.get("reached_goal", False):
                success = True

            state = next_state
            frame = next_frame

            if done:
                break

        # ----------------------------------------------------
        # Episode statistics
        # ----------------------------------------------------

        rewards_history.append(episode_reward)

        if episode_losses:
            avg_loss = float(np.mean(episode_losses))
            losses_history.append(avg_loss)
        else:
            avg_loss = 0.0

        if success:
            successful_episodes += 1

        epsilon = policy.agent.epsilon()

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if episode <= 10 or episode % 10 == 0:

            recent_rewards = rewards_history[-10:]
            recent_avg = np.mean(recent_rewards)

            print(
                f"Episode {episode:4d}/{NUM_EPISODES} | "
                f"Steps: {step + 1:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg10: {recent_avg:7.2f} | "
                f"Loss: {avg_loss:.5f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Success: {'YES' if success else 'NO'}"
            )

        # ----------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------

        if episode % EVAL_EVERY == 0:

            eval_reward, eval_success = evaluate(
                policy,
                start_seed=10000 + episode,
                episodes=EVAL_EPISODES
            )

            eval_success_rate = (
                eval_success / EVAL_EPISODES
            ) * 100.0

            print()
            print("-" * 65)
            print(f"EVALUATION @ Episode {episode}")
            print(f"Average reward : {eval_reward:.3f}")
            print(
                f"Success rate   : "
                f"{eval_success}/{EVAL_EPISODES} "
                f"({eval_success_rate:.1f}%)"
            )
            print("-" * 65)

            # Save best model based on evaluation performance
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

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    overall_success_rate = (
        successful_episodes / NUM_EPISODES
    ) * 100.0

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

    print(
        f"Successful episodes : "
        f"{successful_episodes}/{NUM_EPISODES}"
    )

    print(
        f"Training success    : "
        f"{overall_success_rate:.2f}%"
    )

    print(
        f"Average reward      : "
        f"{average_reward:.3f}"
    )

    print(
        f"Last 10 avg reward  : "
        f"{last_10_average:.3f}"
    )

    print(
        f"Best eval success   : "
        f"{best_success_rate:.1f}%"
    )

    print(
        f"Best eval reward    : "
        f"{best_average_reward:.3f}"
    )

    print()
    print("Best model:")
    print(BEST_MODEL_PATH)

    print()
    print("Final model:")
    print(FINAL_MODEL_PATH)

    print("=" * 65)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(policy, start_seed=10000, episodes=20):

    total_reward = 0.0
    successful = 0
    fail_reasons = {"timeout": 0, "hazard": 0, "enemy": 0, "fell": 0, "other": 0}

    for i in range(episodes):

        seed = start_seed + i

        env.rng = np.random.default_rng(seed)

        state, frame = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            observation = policy.featurizer.transform(
                state,
                None,
                None
            )

            # NO exploration during evaluation
            action_index = policy.agent.select_action(
                observation,
                training=False
            )

            action = Action(action_index)

            next_state, next_frame, reward, done, info = env.step(action)

            episode_reward += reward

            state = next_state
            frame = next_frame

            if info.get("reached_goal", False):
                successful += 1

            if done:
                if not info.get("reached_goal", False):
                    reason = info.get("reason", "timeout" if info.get("timeout") else "other")
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
