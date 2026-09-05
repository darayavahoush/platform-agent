import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


class RolloutBuffer:
    """Stores one on-policy rollout. Cleared after each PPO update."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        obs_dim=71,
        action_dim=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        learning_rate=3e-4,
        update_epochs=4,
        minibatch_size=64,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate
        )

        self.buffer = RolloutBuffer()

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if training:
                action, log_prob, value = self.network.act(state_tensor)
                return (
                    int(action.item()),
                    float(log_prob.item()),
                    float(value.item()),
                )
            else:
                logits, _ = self.network.forward(state_tensor)
                action = torch.argmax(logits, dim=1)
                return int(action.item())

    def remember(self, state, action, log_prob, value, reward, done):
        self.buffer.add(state, action, log_prob, value, reward, done)

    def _compute_gae(self, last_value):
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = [0.0] * len(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[t])
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * next_non_terminal
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = [advantages[t] + values[t] for t in range(len(rewards))]
        return advantages, returns

    def learn(self, last_value):
        if len(self.buffer) == 0:
            return None

        advantages, returns = self._compute_gae(last_value)

        states = torch.tensor(
            np.asarray(self.buffer.states, dtype=np.float32),
            device=self.device,
        )
        actions = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

        # Normalize advantages for stability.
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        num_samples = len(self.buffer)
        indices = np.arange(num_samples)

        total_loss_accum = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.minibatch_size):
                batch_idx = indices[start:start + self.minibatch_size]
                if len(batch_idx) == 0:
                    continue

                batch_idx_t = torch.tensor(
                    batch_idx, dtype=torch.long, device=self.device
                )

                batch_states = states[batch_idx_t]
                batch_actions = actions[batch_idx_t]
                batch_old_log_probs = old_log_probs[batch_idx_t]
                batch_advantages = advantages[batch_idx_t]
                batch_returns = returns[batch_idx_t]

                new_log_probs, entropy, new_values = (
                    self.network.evaluate_actions(batch_states, batch_actions)
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss_accum += float(loss.item())
                num_updates += 1

        self.buffer.reset()

        return total_loss_accum / max(1, num_updates)

    def save(self, path):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
