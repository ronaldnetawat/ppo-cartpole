import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # output probs over actions
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # output state value of the state
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)

        return probs, value


# PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.2, gamma=0.99, lr=3e-4):
        self.epsilon = epsilon
        self.gamma = gamma
        self.policy = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters, lr=lr)

    def select_action(self, state):
        