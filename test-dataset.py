# Adjusted Split of the MNIST Dataset
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download MNIST dataset
full_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the sizes for the three subsets
subset1_size = 9000
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