"""
Independent evaluation of the trained DQN Platform Navigation Agent.

Environment API:

    reset() -> (GameState, Frame)

    step(action) -> (GameState, Frame, reward, done, info)
"""

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

MODEL_PATH = os.path.join(
    "member3_rl",
    "models",
    "dqn_platform_agent_best.pth",
)

NUM_EVAL_EPISODES = 100
MAX_STEPS = 250
BASE_SEED = 20000


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)


# ============================================================
# ENVIRONMENT + POLICY
# ============================================================

env = MockPlatformEnv(seed=BASE_SEED)
policy = DQNPolicy()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

policy.agent.load(MODEL_PATH)


# ============================================================
# STORAGE
# ============================================================

episode_rewards = []
episode_steps = []
episode_success = []

action_counts = {
    action: 0
    for action in Action
}


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("             DQN INDEPENDENT EVALUATION")
print("=" * 65)

print(f"Model        : {MODEL_PATH}")
print(f"Episodes     : {NUM_EVAL_EPISODES}")
print(f"Max steps    : {MAX_STEPS}")
print(f"Base seed    : {BASE_SEED}")

print("=" * 65)


# ============================================================
# EVALUATION LOOP
# ============================================================

for episode in range(1, NUM_EVAL_EPISODES + 1):

    # --------------------------------------------------------
    # Use a different evaluation seed for every episode
    # --------------------------------------------------------

    env.rng = np.random.default_rng(
        BASE_SEED + episode
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # reset() returns GameState + Frame
    # --------------------------------------------------------

    state, frame = env.reset()

    total_reward = 0.0
    steps_taken = 0
    success = False

    # --------------------------------------------------------
    # Run one episode
    # --------------------------------------------------------

    for step in range(1, MAX_STEPS + 1):

        steps_taken = step

        # ----------------------------------------------------
        # Convert GameState -> 64-dimensional observation
        # ----------------------------------------------------

        observation = policy.featurizer.transform(
            state,
            None,
            None,
        )

        # ----------------------------------------------------
        # Select greedy action
        #
        # training=False means:
        # - no epsilon exploration
        # - choose best learned action
        # ----------------------------------------------------

        action_index = policy.agent.select_action(
            observation,
            training=False,
        )

        action = Action(action_index)

        action_counts[action] += 1

        # ----------------------------------------------------
        # STEP ENVIRONMENT
        #
        # Exact API:
        #
        # state, frame, reward, done, info
        # ----------------------------------------------------

        next_state, frame, reward, done, info = env.step(
            action
        )

        # ----------------------------------------------------
        # Accumulate reward
        # ----------------------------------------------------

        total_reward += float(reward)

        # ----------------------------------------------------
        # Check whether goal was reached
        # ----------------------------------------------------

        if info.get("reached_goal", False):
            success = True

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

        # ----------------------------------------------------
        # Stop when environment terminates
        # ----------------------------------------------------

        if done:
            break

    # --------------------------------------------------------
    # Store episode statistics
    # --------------------------------------------------------

    episode_rewards.append(total_reward)
    episode_steps.append(steps_taken)
    episode_success.append(success)

    status = "YES" if success else "NO"

    print(
        f"Episode {episode:3d}/{NUM_EVAL_EPISODES} | "
        f"Steps: {steps_taken:3d} | "
        f"Reward: {total_reward:8.3f} | "
        f"Success: {status}"
    )


# ============================================================
# CONVERT RESULTS TO NUMPY ARRAYS
# ============================================================

episode_rewards = np.asarray(
    episode_rewards,
    dtype=np.float32,
)

episode_steps = np.asarray(
    episode_steps,
    dtype=np.int32,
)

episode_success = np.asarray(
    episode_success,
    dtype=np.bool_,
)


# ============================================================
# CALCULATE STATISTICS
# ============================================================

num_successes = int(
    np.sum(episode_success)
)

success_rate = (
    100.0 * num_successes / NUM_EVAL_EPISODES
)

average_reward = float(
    np.mean(episode_rewards)
)

std_reward = float(
    np.std(episode_rewards)
)

minimum_reward = float(
    np.min(episode_rewards)
)

maximum_reward = float(
    np.max(episode_rewards)
)

average_steps = float(
    np.mean(episode_steps)
)


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 65)
print("                 EVALUATION RESULTS")
print("=" * 65)

print(
    f"Successful episodes : "
    f"{num_successes}/{NUM_EVAL_EPISODES}"
)

print(
    f"Success rate        : "
    f"{success_rate:.2f}%"
)

print(
    f"Average reward      : "
    f"{average_reward:.3f}"
)

print(
    f"Reward std. dev.    : "
    f"{std_reward:.3f}"
)

print(
    f"Minimum reward      : "
    f"{minimum_reward:.3f}"
)

print(
    f"Maximum reward      : "
    f"{maximum_reward:.3f}"
)

print(
    f"Average steps       : "
    f"{average_steps:.2f}"
)


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

print("=" * 65)
print("                  ACTION DISTRIBUTION")
print("=" * 65)

total_actions = sum(
    action_counts.values()
)

for action in Action:

    count = action_counts[action]

    percentage = (
        100.0 * count / total_actions
        if total_actions > 0
        else 0.0
    )

    print(
        f"{action.name:12s} : "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# ASSESSMENT
# ============================================================

print("=" * 65)
print("                    ASSESSMENT")
print("=" * 65)

if success_rate >= 90:

    print(
        "Excellent: The DQN agent achieves a very "
        "high success rate on the independent evaluation."
    )

elif success_rate >= 75:

    print(
        "Good: The DQN agent demonstrates strong "
        "performance, with some failed episodes."
    )

elif success_rate >= 50:

    print(
        "Moderate: The agent has learned useful behavior, "
        "but performance can still be improved."
    )

else:

    print(
        "Needs improvement: The agent is not yet "
        "reliably reaching the goal."
    )


print("=" * 65)
print("Evaluation complete.")
print("=" * 65)