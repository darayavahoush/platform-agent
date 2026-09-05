from core.interfaces import (
    PolicyModule,
    Action,
    GameState,
    Plan,
    PerceptionOutput,
)

from member3_rl.featurizer import GameStateFeaturizer
from member3_rl.dqn import DQNAgent
from member3_rl.ppo import PPOAgent


class DQNPolicy(PolicyModule):

    def __init__(self, obs_dim=71, action_dim=len(Action)):

        self.featurizer = GameStateFeaturizer(
            position_scale=256.0,
            velocity_scale=10.0,
            distance_scale=256.0,
            tick_scale=1000.0,
        )

        self.agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    def act(
        self,
        state: GameState,
        plan: Plan = None,
        perception: PerceptionOutput = None,
        training=True,
    ) -> Action:

        observation = self.featurizer.transform(
            state,
            plan,
            perception,
        )

        action_index = self.agent.select_action(
            observation,
            training=training,
        )

        return Action(action_index)

    def train_step(self, *args, **kwargs):

        loss = self.agent.learn()

        return {
            "loss": loss
        }


class PPOPolicy(PolicyModule):
    def __init__(self, obs_dim=71, action_dim=len(Action)):
        self.featurizer = GameStateFeaturizer(
            position_scale=256.0,
            velocity_scale=10.0,
            distance_scale=256.0,
            tick_scale=1000.0,
        )
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    def act(
        self,
        state: GameState,
        plan: Plan = None,
        perception: PerceptionOutput = None,
    ) -> Action:
        """Interface-compliant action selection (deterministic, greedy).
        Use act_for_rollout() during training to also get log_prob/value."""
        observation = self.featurizer.transform(state, plan, perception)
        action_index = self.agent.select_action(observation, training=False)
        return Action(action_index)

    def act_for_rollout(
        self,
        state: GameState,
        plan: Plan = None,
        perception: PerceptionOutput = None,
    ):
        """Training-time action selection. Returns (Action, log_prob, value,
        observation) so the caller can store the transition for a PPO update."""
        observation = self.featurizer.transform(state, plan, perception)
        action_index, log_prob, value = self.agent.select_action(
            observation, training=True
        )
        return Action(action_index), log_prob, value, observation

    def train_step(self, *args, **kwargs):
        raise NotImplementedError(
            "PPO learns from full rollouts via agent.learn(last_value); "
            "see trainPPO.py for the collection/update loop."
        )