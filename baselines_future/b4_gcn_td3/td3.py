"""TD3 agent (arXiv:2405.05019 Sec. IV-C.4 / Algorithm 1), operating on the
GCN-encoded graph state from `gcn.py`.

Action is 2-dimensional (see `env.py`'s module docstring for why, not the
paper's mn-dimensional per-hop formula): `[accept_logit, offset_frac]`,
both in [-1, 1] via tanh. The caller thresholds `accept_logit` at 0 for the
discrete accept/reject decision and rescales `offset_frac` from [-1, 1]
into `[0, period_ns)` for the flow being decided -- the network itself
never sees `period_ns`, since it isn't part of the graph state.

Design note on the GCN encoder's role: the paper describes GCN training
with its own MSE loss against an unspecified "actual graph representation"
target (Sec. IV-B), which isn't reproducible without more detail than the
paper gives. We instead train ONE shared `GCNStateEncoder` end-to-end as
part of the actor/critic pipeline (standard shared-feature-extractor
actor-critic design) -- gradients from both the critic loss and the
policy gradient flow back through it. A second, target copy of the
encoder is soft-updated alongside the target actor/critics, mirroring
Algorithm 1's phi'/theta1'/theta2' treatment of every other network.

Replay buffer stores raw graph dicts (not fixed-size tensors) since
instances vary in node/edge count -- re-encoded through the GCN on every
sample, online and target alike. Topologies here are small (well under
200 queue-level nodes even at 16 switches), so this costs milliseconds
per graph, not a bottleneck at the batch sizes TD3 uses.
"""
from __future__ import annotations

import copy
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_future.b4_gcn_td3.gcn import GCNStateEncoder

ACTION_DIM = 2  # [accept_logit, offset_frac], both tanh-squashed to [-1, 1]


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TwinCritic(nn.Module):
    """Both Q1 and Q2 in one module -- Algorithm 1 always updates and
    queries them together (min of the two for targets, both for the
    critic loss)."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + ACTION_DIM
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, graph: dict, action, reward: float, next_graph: dict, done: bool) -> None:
        self.buffer.append((graph, action, reward, next_graph, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class TD3Agent:
    """Implements Algorithm 1 exactly (initialize, target soft-updates
    every eta steps by tau, twin-critic min-target, delayed policy
    updates) -- hyperparameter defaults follow Fujimoto et al. 2018
    (arXiv's own [24, Tab. 3] reference), since the paper states it
    aligned its own hyperparameters with that source rather than listing
    them itself."""

    def __init__(
        self,
        graph_embedding_dim: int = 32,
        hidden: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,   # eta
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self._update_count = 0

        self.encoder = GCNStateEncoder(graph_embedding_dim=graph_embedding_dim).to(self.device)
        self.actor = Actor(graph_embedding_dim, hidden).to(self.device)
        self.critic = TwinCritic(graph_embedding_dim, hidden).to(self.device)

        self.encoder_target = copy.deepcopy(self.encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for net in (self.encoder_target, self.actor_target, self.critic_target):
            for p in net.parameters():
                p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()), lr=critic_lr,
        )

    def select_action(self, graph: dict, explore: bool = True) -> torch.Tensor:
        with torch.no_grad():
            state = self.encoder(graph)
            action = self.actor(state)
            if explore:
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(-1.0, 1.0)
            return action.cpu()

    def _batch_states(self, encoder: nn.Module, graphs: list[dict]) -> torch.Tensor:
        return torch.stack([encoder(g) for g in graphs])

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        if len(buffer) < batch_size:
            return {}
        batch = buffer.sample(batch_size)
        graphs, actions, rewards, next_graphs, dones = zip(*batch)
        actions = torch.stack([a if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float32)
                                for a in actions]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_states = self._batch_states(self.encoder_target, list(next_graphs))
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.min(target_q1, target_q2)

        states = self._batch_states(self.encoder, list(graphs))
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logs = {"critic_loss": critic_loss.item()}
        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            # Re-encode with the (just-updated) online encoder for the
            # policy gradient -- states above were used for the critic
            # step and their graph would otherwise be stale.
            states_for_actor = self._batch_states(self.encoder, list(graphs))
            actor_loss = -self.critic.q1_only(states_for_actor, self.actor(states_for_actor)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            logs["actor_loss"] = actor_loss.item()

            self._soft_update(self.encoder, self.encoder_target)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        return logs

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for p, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(), "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
