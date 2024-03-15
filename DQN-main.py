import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import deque
import random
import math
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
# CNN for feature extraction
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
feature_dim = 10  # Output dim of CNN
state_dim = feature_dim * 20  # Concatenated features of 20 images
action_dim = 20  # 20 images to choose from
learning_rate = 0.001
batch_size = 20

# Initialize models
feature_extractor = CNN()
dqn_model = DQN(state_dim, action_dim)
optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset = datasets.MNIST('dataset/', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
replay_buffer_capacity = 10000
replay_buffer = ReplayBuffer(replay_buffer_capacity)
target_update_frequency = 10
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 200

target_dqn = DQN(state_dim, action_dim)
target_dqn.load_state_dict(dqn_model.state_dict())

def epsilon_by_frame(frame_idx):
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * frame_idx / epsilon_decay)


# Function to get the next state and reward
def get_next_state_and_reward(current_state, action, label, new_image_feature):
    # Calculate reward based on MSE
    reward = loss_fn(label, feature_extractor(action.unsqueeze(0)).squeeze(0)).item()
    # Update state by removing selected image and adding new image
    next_state = torch.cat((current_state[feature_dim:], new_image_feature))
    return next_state, reward

def compute_loss(batch, gamma=0.999):
    states, actions, rewards, next_states = zip(*batch)
    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)

    # Current Q values
    current_q_values = dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q values
    next_q_values = target_dqn(next_states).max(1)[0].detach()

    # Expected Q values
    expected_q_values = rewards + gamma * next_q_values

    # MSE Loss
    loss = nn.MSELoss()(current_q_values, expected_q_values)
    return loss




# Training loop (simplified)
num_epochs = 10
frame_idx = 0
for epoch in range(num_epochs):
    for images, labels in data_loader:
        image_features = feature_extractor(images).view(batch_size, -1)
        state = image_features[:20].flatten()

        for i in range(20, batch_size):
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)

            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    action_idx = dqn_model(state.unsqueeze(0)).max(1)[1].item()
            else:
                action_idx = random.randint(0, action_dim - 1)

            action = image_features[action_idx]
            next_state, reward = get_next_state_and_reward(state, action, labels[action_idx], image_features[i])

            # Store in replay buffer
            replay_buffer.push(state, action_idx, reward, next_state)

            # Update state
            state = next_state

            # Start training once buffer has a certain number of samples
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                # Unpack batch and compute loss
                # Update DQN model (not shown in this snippet)

            # Update target network
            if frame_idx % target_update_frequency == 0:
                target_dqn.load_state_dict(dqn_model.state_dict())

            # During training loop
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


