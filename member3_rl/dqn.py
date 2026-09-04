import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=150000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim=64,
        action_dim=6,
        gamma=0.99,
        learning_rate=2.5e-4,
        batch_size=64,
        target_update=1000,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=150000,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size

        self.target_update = target_update

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.learn_steps = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_network = QNetwork(
            obs_dim,
            action_dim
        ).to(self.device)

        self.target_network = QNetwork(
            obs_dim,
            action_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.replay_buffer = ReplayBuffer()

    def epsilon(self):
        fraction = min(
            1.0,
            self.learn_steps / self.epsilon_decay
        )

        return (
            self.epsilon_start
            + fraction *
            (self.epsilon_end - self.epsilon_start)
        )

    def select_action(self, state, training=True):

        if training and random.random() < self.epsilon():
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        current_q = self.q_network(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN:
        # online network chooses next action
        next_actions = self.q_network(
            next_states
        ).argmax(
            dim=1,
            keepdim=True
        )

        # target network evaluates it
        next_q = self.target_network(
            next_states
        ).gather(
            1,
            next_actions
        ).squeeze(1)

        target_q = rewards + (
            1.0 - dones
        ) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q.detach()
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            10.0
        )

        self.optimizer.step()

        self.learn_steps += 1

        if self.learn_steps % 500 == 0:
            avg_q = current_q.mean().item()
            print(f"    [diag] learn_step={self.learn_steps} avg_q={avg_q:.3f} loss={loss.item():.4f}")

        if self.learn_steps % self.target_update == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.q_network.load_state_dict(
            checkpoint["q_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.learn_steps = checkpoint.get(
            "learn_steps",
            0
        )