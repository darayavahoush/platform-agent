from core.interfaces import (
    PolicyModule,
    Action,
    GameState,
    Plan,
    PerceptionOutput,
)

from member3_rl.featurizer import GameStateFeaturizer
from member3_rl.dqn import DQNAgent


class DQNPolicy(PolicyModule):

    def __init__(self, obs_dim=65, action_dim=len(Action)):

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