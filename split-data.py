import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST('dataset/', train=True, download=False, transform=transform)

# Extract features (images) and labels from the dataset
data = train_dataset.data.numpy()
labels = train_dataset.targets.numpy()

# Split the dataset into three parts (60% training, 20% validation, 20% test)
x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Further split the training set into three parts
x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
x_train2, x_train3, y_train2, y_train3 = train_test_split(x_train2, y_train2, test_size=0.5, random_state=42)

# Create PyTorch datasets for training, validation, and testing
train_dataset1 = torch.utils.data.TensorDataset(torch.Tensor(x_train1).view(-1, 1, 28, 28), torch.LongTensor(y_train1))
train_dataset2 = torch.utils.data.TensorDataset(torch.Tensor(x_train2).view(-1, 1, 28, 28), torch.LongTensor(y_train2))
train_dataset3 = torch.utils.data.TensorDataset(torch.Tensor(x_train3).view(-1, 1, 28, 28), torch.LongTensor(y_train3))
val_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_val).view(-1, 1, 28, 28), torch.LongTensor(y_val))
test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_test).view(-1, 1, 28, 28), torch.LongTensor(y_test))

# Create PyTorch data loaders
train_loader1 = DataLoader(train_dataset1, batch_size=64, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=64, shuffle=True)
train_loader3 = DataLoader(train_dataset3, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print the sizes of the resulting datasets
print("Training set 1 size:", len(train_loader1.dataset))
print("Training set 2 size:", len(train_loader2.dataset))
print("Training set 3 size:", len(train_loader3.dataset))
print("Validation set size:", len(val_loader.dataset))
print("Test set size:", len(test_loader.dataset))