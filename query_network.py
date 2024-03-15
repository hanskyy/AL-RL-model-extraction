import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.action_space = n_actions

        # Example architecture
        self.feature = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Ensuring advantages is two-dimensional, [batch_size, number_of_actions]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(0)  # Fix for single example
        # Compute Q values
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals