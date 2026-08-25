"""Member 3: PPO/DQN policy that learns navigation/control, optionally
conditioned on Plan (from Member 2) and PerceptionOutput/embedding (from
Member 1). Must generalize to unseen levels -> train across a level
distribution, not a single level. Implement PolicyModule from core.interfaces.
"""
from core.interfaces import PolicyModule, Action, GameState, Plan, PerceptionOutput
from typing import Optional


class PPOPolicy(PolicyModule):
    def __init__(self, obs_dim: int = 64, action_dim: int = len(Action)):
        # TODO: actor-critic network. Obs = symbolic GameState features
        # (+ optional perception.embedding) (+ plan.waypoints/risk as aux input).
        pass

    def act(self, state: GameState, plan: Optional[Plan] = None,
            perception: Optional[PerceptionOutput] = None) -> Action:
        raise NotImplementedError

    def train_step(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class DQNPolicy(PolicyModule):
    """Baseline for comparison against PPO; simpler discrete-action off-policy learner."""
    def __init__(self, obs_dim: int = 64, action_dim: int = len(Action)):
        pass

    def act(self, state: GameState, plan: Optional[Plan] = None,
            perception: Optional[PerceptionOutput] = None) -> Action:
        raise NotImplementedError

    def train_step(self, *args, **kwargs) -> dict:
        raise NotImplementedError
