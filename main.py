import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from query_network import DQN, DuelingDQN
import math

target_update_frequency = 10
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 200
# Define ReplayBuffer for store transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Define the CNN architecture (this should match the architecture used for training)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1000) # 28x28 pixels divided by 2 due to pooling
        self.fc2 = nn.Linear(1000, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        # Apply convolutions, activation function (ReLU), and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the image for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # Apply the fully connected layers with ReLU and output layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FeatureExtractionCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)  # Reduced channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # Reduced channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Additional pooling layer
        self.fc = nn.Linear(16 * 7 * 7, 100)  # Fully connected layer for dimensionality reduction

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)  # Flatten
        x = F.relu(self.fc(x))  # Dimensionality reduction
        return x

def sample_images(data_loader, num_samples):
    """
    Randomly samples 'num_samples' images (and their labels) from a given DataLoader.

    Args:
    data_loader (DataLoader): The DataLoader to sample from.
    num_samples (int): Number of samples to retrieve.

    Returns:
    (torch.Tensor, torch.Tensor): Tuple of images and labels.
    """
    images, labels = [], []
    all_indices = list(range(len(data_loader.dataset)))
    sampled_indices = random.sample(all_indices, num_samples)

    for idx in sampled_indices:
        image, label = data_loader.dataset[idx]
        images.append(image)
        labels.append(label)

    # Stack the list of images and labels into tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels

# Example usage:
# sampled_images, sampled_labels = sample_images(train_loader2, 20)
def extract_and_concatenate_features(images, model):
    """
    Extracts features from a batch of images using the provided CNN model and
    concatenates them into a single vector.

    Args:
    images (torch.Tensor): A batch of images, typically of shape [batch_size, channels, height, width].
    model (nn.Module): The CNN model used for feature extraction.

    Returns:
    torch.Tensor: A single concatenated feature vector.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model and images to the selected device
    model = model.to(device)
    images = images.to(device)
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():  # No need to track gradients here
        # Extract features for the batch of images
        extracted_features = model(images)  # Shape: [batch_size, feature_dim]

        # Flatten and concatenate features from all images into a single vector
        concatenated_features = torch.flatten(extracted_features, start_dim=0)

    return concatenated_features

def select_action(model, state, epsilon=0.1):
    """
    Selects an action using the DQN model based on the current state.

    Args:
    model (nn.Module): The DQN model used for action selection.
    state (torch.Tensor): The current state represented as a concatenated feature vector.
    epsilon (float): The probability of selecting a random action (exploration).

    Returns:
    int: The index of the selected image.
    torch.Tensor, torch.Tensor: The selected image and its corresponding label.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the model and state are on the correct device
    model = model.to(device)
    state = state.to(device)

    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # Model is used to predict the action
            model.eval()  # Set the model to evaluation mode
            #state = state.unsqueeze(0)  # Add a batch dimension (batch size of 1)
            action_scores = model(state)
            action = torch.argmax(action_scores).item()
    else:
        # Randomly select an action
        action = random.randrange(20)

    return action

def calculate_mse_reward(trained_cnns, selected_image, selected_label):
    """
    Calculates the mean squared error reward based on the predictions of the CNN models.

    Args:
    cnn_models (list of nn.Module): List of trained CNN models.
    selected_image (torch.Tensor): The selected image.
    selected_label (int): The actual label of the selected image.

    Returns:
    float: The mean MSE reward from all models.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_losses = []

    # Convert selected_label to one-hot encoding
    selected_image = selected_image.to(device)
    label_one_hot = F.one_hot(torch.tensor(selected_label), num_classes=10).float().to(device)

    # Expand dimensions of selected_image to match model input
    # Assuming selected_image is of shape [1, 28, 28], needs to be [1, 1, 28, 28]
    if selected_image.dim() == 3:
        selected_image = selected_image.unsqueeze(0)

    for model in trained_cnns:
        model = model.to(device)
        # Forward pass to get the logits
        with torch.no_grad():
            model.eval()
            logits = model(selected_image)

        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)

        # Calculate MSE loss
        mse_loss = F.mse_loss(probabilities, label_one_hot.unsqueeze(0))
        mse_losses.append(mse_loss.item())

    # Average the MSE losses from all models
    mean_mse_reward = - sum(mse_losses) / len(mse_losses)

    return mean_mse_reward
def update_dqn(model, target_model, optimizer, criterion, minibatch, gamma=0.99):
    """
    Updates the DQN based on a minibatch of experiences, using a target Q-network.

    Args:
    model (nn.Module): The main DQN model.
    target_model (nn.Module): The target Q-network, with stable weights.
    optimizer (torch.optim.Optimizer): The optimizer for the DQN.
    criterion (nn.Module): Loss function.
    minibatch (list of tuples): Each tuple is (state, action, reward, next_state).
    gamma (float): Discount factor for future rewards.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move models to the selected device
    model = model.to(device)
    target_model = target_model.to(device)

    states, actions, rewards, next_states = zip(*minibatch)

    # Convert to PyTorch tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(device)

    # Compute Q(s_t, a) using the main model
    model.eval()
    state_action_values = model(states).gather(1, actions.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states using the target model
    next_state_values = torch.zeros(len(minibatch), device=device)
    if len(non_final_next_states) > 0:
        next_state_values = target_model(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + rewards

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def update_double_dqn(model, target_model, optimizer, criterion, minibatch, gamma=0.99):
    """
    Updates the DQN based on a minibatch of experiences, using Double DQN approach.
    In Double DQN, the selection of action in the next state is done by the current model,
    and the evaluation of that action is done by the target model.

    Args:
    model (nn.Module): The main DQN model.
    target_model (nn.Module): The target Q-network, with stable weights.
    optimizer (torch.optim.Optimizer): The optimizer for the DQN.
    criterion (nn.Module): Loss function.
    minibatch (list of tuples): Each tuple is (state, action, reward, next_state, done).
    gamma (float): Discount factor for future rewards.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move models to the selected device
    model = model.to(device)
    target_model = target_model.to(device)

    states, actions, rewards, next_states = zip(*minibatch)

    # Convert to PyTorch tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool, device=device)
    non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    model.eval()
    state_action_values = model(states).gather(1, actions.unsqueeze(1))

    # Initialize the next state values to zero for all states, and we will overwrite these
    # values for all non-terminal states below
    next_state_values = torch.zeros(len(minibatch), device=states.device)

    if len(non_final_next_states) > 0:
        # Here is the major difference between DQN and Double DQN:
        # 1. Select the best action in next states (action selection) from the current model
        model.eval()  # Make sure the model is in eval mode for inference
        best_actions = model(non_final_next_states).max(1)[1].detach()

        # 2. Evaluate the selected action with the target model (action evaluation)
        target_model.eval()  # Make sure the target model is in eval mode for inference
        next_state_values[non_final_mask] = target_model(non_final_next_states).gather(1, best_actions.unsqueeze(
            1)).squeeze().detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + rewards

    # Compute loss
    model.train()  # Switch the model back to training mode before the backward pass
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def epsilon_by_frame(frame_idx):
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * frame_idx / epsilon_decay)
def load_trained_cnn_model(model_path):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model
# Assuming MNIST_CNN and DQN classes are defined
# MNIST_CNN - CNN model for feature extraction
# DQN - Deep Q-Network model
def retrain_cnn_models(trained_cnns, optimizers, selected_image, selected_label):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the data to the selected device
    selected_image = selected_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    selected_label = torch.tensor([selected_label], device=device)  # Ensure label is a tensor and move to device

    for model, optimizer in zip(trained_cnns, optimizers):
        # Move model to the same device
        model.to(device)

        model.train()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(selected_image)
        loss = F.cross_entropy(outputs, selected_label)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    # Function to get the next state
def get_next_state(current_state, new_image_feature):
    next_state = np.concatenate((current_state[1:], [new_image_feature]))
    return next_state
def soft_update(target_network, online_network, tau=0.001):
    """
    Softly updates the target network weights based on the online network weights.

    Parameters:
    - target_network: The target network (PyTorch model)
    - online_network: The online network (PyTorch model)
    - tau: The interpolation parameter indicating how much of the online network's
           weights to blend into the target network's weights.
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Download MNIST dataset
full_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the sizes for the three subsets
subset1_size = 1000
subset2_size = 21000
subset3_size = len(full_train_set) - subset1_size - subset2_size  # Remaining for dataset3

# Randomly split the dataset into three subsets
train_subset1, train_subset2, remaining_subset = random_split(full_train_set, [subset1_size, subset2_size, subset3_size])

# Creating DataLoaders for each subset
train_loader1 = DataLoader(train_subset1, batch_size=64, shuffle=True)
train_loader2 = DataLoader(train_subset2, batch_size=64, shuffle=True)
train_loader3 = DataLoader(remaining_subset, batch_size=64, shuffle=True) # This will be used as the third dataset

# Test DataLoader remains the same
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Hyperparameters
# num_episodes = 1000
max_steps_per_episode = 4
feature_size = 2000
replay_buffer_capacity = 10000
batch_size = 20
frame_idx = 0
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(3):
    num_episodes = (i + 1) * 1000 + 2000
    # Initialize the DQN
    dqn = DQN(feature_size, 20)  # Adjust sizes as needed
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialize target DQN
    target_dqn = DQN(feature_size, 20)
    target_dqn.load_state_dict(dqn.state_dict())

    # Experience replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    feature_extraction_model = FeatureExtractionCNN()
    dqn.to(device)
    target_dqn.to(device)
    # Load the 10 trained models
    trained_cnns = [load_trained_cnn_model(f'./mnist_model_{l}.pth') for l in range(10)]
    # Assuming cnn_models is a list of your 10 CNN models
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in trained_cnns]
    # Training Loop
    for episode in range(num_episodes):
        # Initialize state - Randomly pick 20 images and extract features
        initial_images, initial_labels = sample_images(train_loader2, 20)

        state = extract_and_concatenate_features(initial_images, feature_extraction_model)

        for t in range(1, max_steps_per_episode):
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            # Select an action (image) - either random or using the policy network
            action = select_action(dqn, state, epsilon)
            # Retrieve the selected image and label
            selected_image = initial_images[action]
            selected_label = initial_labels[action]
            # Get next state - Replace one image with a new one
            new_image, new_label = sample_images(train_loader2, 1)
            initial_images[action] = new_image
            initial_labels[action] = new_label
            next_state = extract_and_concatenate_features(initial_images, feature_extraction_model)

            # Calculate reward - MSE between CNN predictions and label
            # In the future I will put original model to replace selected_label
            reward = calculate_mse_reward(trained_cnns, selected_image, selected_label)

            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state)


            # Update target network
            # if frame_idx % target_update_frequency == 0:
            #     target_dqn.load_state_dict(dqn.state_dict())
            soft_update(target_dqn, dqn)
            # Update DQN with the minibatch
            if len(replay_buffer) > batch_size:
                minibatch = replay_buffer.sample(batch_size)
                update_dqn(dqn, target_dqn, optimizer, criterion, minibatch)

            # Update state
            state = next_state

            # Optionally retrain CNN models here
            retrain_cnn_models(trained_cnns, optimizers, selected_image, selected_label)

        # End of episode
    # Save the trained DQN model
    torch.save(dqn.state_dict(),f'./softdqn_model{num_episodes}.pth')