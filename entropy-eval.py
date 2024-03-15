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
from query_network import DQN
import math
from torch.autograd import Variable
from skimage.color import rgb2gray
from scipy.stats import entropy
from skimage import data

target_update_frequency = 10
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 200
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
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():  # No need to track gradients here
        # Extract features for the batch of images
        extracted_features = model(images)  # Shape: [batch_size, feature_dim]

        # Flatten and concatenate features from all images into a single vector
        concatenated_features = torch.flatten(extracted_features, start_dim=0)

    return concatenated_features

def select_action(model, state, epsilon=0):
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
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # Model is used to predict the action
            model.eval()  # Set the model to evaluation mode
            action_scores = model(state)
            action = torch.argmax(action_scores).item()
    else:
        # Randomly select an action
        action = random.randrange(20)

    return action

def epsilon_by_frame(frame_idx):
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * frame_idx / epsilon_decay)

def load_trained_cnn_model(model_path):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


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

def get_next_state(current_state, new_image_feature):
    next_state = np.concatenate((current_state[1:], [new_image_feature]))
    return next_state


def calculate_image_entropy(image):
    """Calculate the entropy of an image."""
    # Adjust for image tensors with shape (1, height, width)
    if image.shape[0] == 1:
        # Remove the first dimension
        image = image.squeeze(0)

    # Normalize pixel values if they aren't already (assuming they're in the range 0-255)
    if image.max() > 1.0:
        image = image / 255.0

    # Calculate the histogram of pixel values
    histogram, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
    histogram = histogram[histogram > 0]  # Remove zeros to avoid log(0) in entropy calculation
    # Calculate and return the entropy
    return entropy(histogram)

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
num_episodes = 1000
max_steps_per_episode = 1
feature_size = 2000
replay_buffer_capacity = 10000
batch_size = 20
frame_idx = 0

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('model_accuracies.txt','w') as file:
    for k in range(3):
        num = (k + 1) * 1000 + 2000

        criterion = nn.MSELoss()
        feature_extraction_model = FeatureExtractionCNN()

        # Load the 10 trained models
        trained_cnns_random = [load_trained_cnn_model(f'./mnist_model_{i}.pth') for i in range(10)]
        trained_cnns_dqn = [load_trained_cnn_model(f'./mnist_model_{i}.pth') for i in range(10)]
        # Assuming cnn_models is a list of your 10 CNN models
        optimizers_random = [optim.Adam(model.parameters(), lr=0.001) for model in trained_cnns_random]
        optimizers_dqn = [optim.Adam(model.parameters(), lr=0.001) for model in trained_cnns_dqn]

        # Train 10 times
        random_a = []
        dqn_a = []
        for j in range(10):

            # Train 10 CNN models
            for episode in range(num_episodes):
                # Initialize state - Randomly pick 20 images and extract features
                initial_images, initial_labels = sample_images(train_loader3, 20)

                # Calculate entropy for each image
                entropies = [calculate_image_entropy(image) for image in initial_images]
                action = np.argmax(entropies)
                # Retrieve the selected image and label
                selected_image = initial_images[action]
                selected_label = initial_labels[action]

                # Optionally retrain CNN models here
                retrain_cnn_models(trained_cnns_dqn, optimizers_dqn, selected_image, selected_label)

                action = random.randrange(20)
                selected_image = initial_images[action]
                selected_label = initial_labels[action]
                retrain_cnn_models(trained_cnns_random, optimizers_random, selected_image, selected_label)
                # End of episode
            # print("Finished training\n")

            # print("Mean of 10 models accuracy is {:.0f}".format(mean_of_numbers))
            # Test Models based on random
            i = 0
            accuracy = []
            for model, optimizer in zip(trained_cnns_random, optimizers_random):
                model.eval()  # evaluation/testing
                model.to(device)  # Move model to the selected device (GPU or CPU)
                test_loss = 0
                correct = 0
                for data, label in test_loader:  # separate data and label
                    data, label = data.to(device), label.to(device)
                    output = model(data)  # enter data into model, save in output
                    test_loss += F.nll_loss(output, label, size_average=False).data  #
                    pred = output.data.max(1, keepdim=True)[1]  # prediction result
                    correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # if label=pred then correct++
                test_loss /= len(test_loader.dataset)  # compute test loss
                # print('\nAverage Loss: {:.4f}, Accuracy: {:.0f}'.format(test_loss, 100. * correct / len(test_loader.dataset)))
                accuracy.append(100. * correct / len(test_loader.dataset))
                # print(f'Model {i + 1} tested.')
                i = i + 1
            # print("All models are tested based on random pick.")
            mean_of_numbers = sum(accuracy) / len(accuracy)
            random_a.append(mean_of_numbers)
            # print("Mean of 10 models accuracy is {:.0f}".format(mean_of_numbers))
            # Test Models based on DQN
            i = 0
            accuracy = []
            for model, optimizer in zip(trained_cnns_dqn, optimizers_dqn):
                model.eval()  # evaluation/testing
                model.to(device)  # Move model to the selected device (GPU or CPU)
                test_loss = 0
                correct = 0
                for data, label in test_loader:  # separate data and label
                    data, label = data.to(device), label.to(device)
                    output = model(data)  # enter data into model, save in output
                    test_loss += F.nll_loss(output, label, size_average=False).data  #
                    pred = output.data.max(1, keepdim=True)[1]  # prediction result
                    correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # if label=pred then correct++
                test_loss /= len(test_loader.dataset)  # compute test loss
                # print('\nAverage Loss: {:.4f}, Accuracy: {:.0f}'.format(test_loss, 100. * correct / len(test_loader.dataset)))
                accuracy.append(100. * correct / len(test_loader.dataset))
                # print(f'Model {i + 1} tested.')
                i = i + 1
            # print("All models are tested based on DQN.")
            mean_of_numbers = sum(accuracy) / len(accuracy)
            dqn_a.append(mean_of_numbers)
            # print("Mean of 10 models accuracy is {:.0f}".format(mean_of_numbers))

        # print(f"\nThe model {num:.0f}k3 random accuracy is {random_a:.0f}%")
        # print(f"The model {num:.0f}k3 dqn accuracy is {dqn_a:.0f}%")
        formatted_random_a = ', '.join(f"{tensor.item():.4f}%" for tensor in random_a)
        text1 = f"The model {num:.0f} random accuracy is {formatted_random_a}%"
        formatted_dqn_a = ', '.join(f"{tensor.item():.4f}%" for tensor in dqn_a)
        text2 = f"The model {num:.0f} entropy accuracy is {formatted_dqn_a}%"

        file.write(text1 + "\n" + text2 + "\n")