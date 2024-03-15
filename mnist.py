import torch #import torch library
import torch.nn as nn #import torch neural network library
import torch.nn.functional as F #import functional neural network module
import torch.optim as optim #import optimizer neural network module
from torch.autograd import Variable #import variable that connect to automatic differentiation
from torchvision import datasets, transforms #import torchvision for datasets and transform
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__() #load super class for training data
		self.fc1 = nn.Linear(784, 320)
		self.fc2 = nn.Linear(320, 50)
		self.fc3 = nn.Linear(50, 10)
		self.relu = nn.ReLU()
	
	def forward(self, x): #feed forward
		layer1 = x.view(-1, 784) #make it flat from 0 - 320
		layer2 = self.relu(self.fc1(layer1)) #layer2 = layer1 -> fc1 -> relu
		layer3 = self.relu(self.fc2(layer2)) #layer3 = layer2 -> fc2 -> relu
		layer4 = self.relu(self.fc3(layer3)) #layer4 = layer3 -> fc2 -> relu
		return F.log_softmax(layer4) #softmax activation to layer4


class CNN(nn.Module): #class model
	def __init__(self):
		super(CNN, self).__init__() #load super class for training data
		self.conv1 = nn.Conv2d(1, 10, 5) #Convolutional modul: input, output, kernel
		self.conv2 = nn.Conv2d(10, 20, 5) #Convolutional modul: input, output, kernel
		self.maxpool = nn.MaxPool2d(2) #maxpooling modul: kernel
		self.relu = nn.ReLU() #activation relu modul
		self.dropout2d = nn.Dropout2d() #dropout modul
		self.dropout = nn.Dropout() #dropout modul
		self.fc1 = nn.Linear(320, 50) #Fully Connected modul: input, output
		self.fc2 = nn.Linear(50, 10)# Fully Connected modul: input, output

	def forward(self, x): #feed forward
		layer1 = self.relu(self.maxpool(self.conv1(x))) # layer1 = x -> conv1 -> maxpool -> relu
		layer2 = self.relu(self.maxpool(self.dropout2d(self.conv2(layer1)))) #layer2 = layer1 -> conv2 -> dropout -> maxpool -> relu
		layer3 = layer2.view(-1, 320) #make it flat from 0 - 320
		layer4 = self.relu(self.fc1(layer3)) #layer4 = layer3 -> fc1 -> relu
		layer5 = self.fc2(layer4) #layer5 = layer4 -> fc2
		return F.log_softmax(layer5) #softmax activation to layer5

class Dataset:
	def read(self):
		#load train and test loader that will be normalized and shuffle
		train_loader = torch.utils.data.DataLoader( 
			datasets.MNIST('dataset/',train=True, download=True,
				 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
				 batch_size=1000, shuffle=True)
		test_loader = torch.utils.data.DataLoader(
			datasets.MNIST('dataset/',train=False, 
				 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
				 batch_size=1000, shuffle=True)
		return train_loader, test_loader

class CustomMNISTLoader:
    def __init__(self):
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
		train_subset1, train_subset2, remaining_subset = random_split(full_train_set,
																	  [subset1_size, subset2_size, subset3_size])

		# Creating DataLoaders for each subset
		self.train_loader1 = DataLoader(train_subset1, batch_size=64, shuffle=True)
		self.train_loader2 = DataLoader(train_subset2, batch_size=64, shuffle=True)
		self.train_loader3 = DataLoader(remaining_subset, batch_size=64,
								   shuffle=True)  # This will be used as the third dataset

		# Test DataLoader remains the same
		self.test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    def read(self):
        return self.train_loader1, self.test_loader

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode',required=True,help="cnn / dnn")
	parser.add_argument('-e','--epoch',required=False,help="")
	args = vars(parser.parse_args())
	if(args['mode']=="cnn"):
		model = CNN() #load graph / model
	if(args['mode']=='dnn'):
		model = DNN()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #load optimizer SGD with momentum 0.9 and learning rate 0.01
	train_loader,test_loader = CustomMNISTLoader().read()
	for epoch in range(int(args['epoch'])): # train epoch = 10
		model.train() #training
		for batch_idx, (data, label) in enumerate(train_loader): #enumerate train_loader per batch-> index, (data, label) ex: 0, (img1, 4)... 1, (img2, 2)
			data, label = Variable(data), Variable(label) #create torch variable and enter each data and label into it
			optimizer.zero_grad()
			output = model(data) #enter data into model, save in output
			train_loss = F.nll_loss(output, label) #nll = negative log likehood loss between output and label. it useful for classification problem with n class
			train_loss.backward() #compute gradient
			optimizer.step() #update weight
			if batch_idx % 10 == 0: #display step
				print('Train Epochs: {}, Loss: {:.6f} '.format(epoch, train_loss.data )) #print
		model.eval() #evaluation/testing
		test_loss = 0
		correct = 0
		for data, label in test_loader: #separate data and label
			data, label = Variable(data,volatile=True), Variable(label) #create torch variable and enter data and label into it
			output = model(data) #enter data into model, save in output
			test_loss += F.nll_loss(output, label, size_average=False).data #
			pred = output.data.max(1, keepdim=True)[1] #prediction result
			correct += pred.eq(label.data.view_as(pred)).cpu().sum() #if label=pred then correct++
		test_loss /= len(test_loader.dataset) #compute test loss
		print('\nAverage Loss: {:.4f}, Accuracy: {:.0f}'.format(test_loss,  100. * correct / len(test_loader.dataset)))
	model_scripted = torch.jit.script(model)  # Export to TorchScript
	model_scripted.save('model_scripted.pt')  # Save