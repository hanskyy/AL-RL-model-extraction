# Full Code for Training and Saving 10 CNN Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

# Step 2: CNN Architecture (as defined previously)
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

# [CNN class definition remains the same as before]

# Step 3: Training Function (as defined previously)
def train_model(model, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# [Training function code remains the same as before]

# Step 4: Model Initialization and Training
# Load and Split MNIST dataset
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

# Train and Save Models

for i in range(10):
    model = MNIST_CNN()
    model.train()
    print(f"Training Model {i + 1}")
    trained_model = train_model(model, train_loader1)

    model.eval()  # evaluation/testing
    test_loss = 0
    correct = 0
    for data, label in test_loader:  # separate data and label
        data, label = Variable(data, volatile=True), Variable(
            label)  # create torch variable and enter data and label into it
        output = model(data)  # enter data into model, save in output
        test_loss += F.nll_loss(output, label, size_average=False).data  #
        pred = output.data.max(1, keepdim=True)[1]  # prediction result
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # if label=pred then correct++
    test_loss /= len(test_loader.dataset)  # compute test loss
    print('\nAverage Loss: {:.4f}, Accuracy: {:.0f}'.format(test_loss, 100. * correct / len(test_loader.dataset)))
    # Save each model
    torch.save(trained_model.state_dict(), f'./mnist_model_{i}.pth')
    print(f'Model {i + 1} trained and saved.')
print("All models are trained and saved.")
