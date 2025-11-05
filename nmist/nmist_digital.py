import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import torch  # the library
import torch.nn as nn  # neural network modules - CNNs, RNNs, Linear
import torch.nn.functional as F  # Activation Functions
import torch.optim as optim  # optimizers like SGD, Adam etc
from torch.utils.data import DataLoader, TensorDataset  # helps load data, make batches


def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Define file paths
base_path = "data/nmist"  # Adjust this path to your actual dataset
print(os.listdir(base_path))

train_images = load_images(os.path.join(base_path, "train-images.idx3-ubyte"))
train_labels = load_labels(os.path.join(base_path, "train-labels.idx1-ubyte"))

test_images = load_images(os.path.join(base_path, "t10k-images.idx3-ubyte"))
test_labels = load_labels(os.path.join(base_path, "t10k-labels.idx1-ubyte"))

# Confirm shapes
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")

plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.axis('off')
plt.show()

# https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy
train_images_tensor = torch.from_numpy(train_images).float() / 255.0
train_images_tensor = train_images_tensor.unsqueeze(1)  # unsqueezes the channel which is required by CNNS
train_labels_tensor = torch.from_numpy(train_labels).long()
test_images_tensor = torch.from_numpy(test_images).float() / 255.0
test_images_tensor = test_images_tensor.unsqueeze(1)
test_labels_tensor = torch.from_numpy(test_labels).long()

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)


class NN(
    nn.Module):  # a new class NN that inherits from nn.Module (which is PyTorch’s base class for all neural networks)
    def __init__(self, input_size, num_classes):  #
        super(NN, self).__init__()  # initialization, super() gives you access to methods in a parent class
        self.fc1 = nn.Linear(input_size, 50)  # Layer 1 input-> 50 nodes
        self.fc2 = nn.Linear(50, num_classes)  # Layer 2 50-> num of classes

    def forward(self, x):
        # would run on input x
        x = self.fc1(x)  # apply 1st linear tranformation
        x = F.relu(x)  # apply activation
        x = self.fc2(x)  # apply 2nd tranformation
        return x


model = NN(28 * 28, 10)  # image pixel, 10 number of digits
x = torch.randn(64, 28 * 28)
print(model(x).shape)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28*28
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 4

# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# N: Number of images
# C: channels - grayscale = 1 channel
# if colored  = rgb, channels = 3

model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train NW
for epoch in range(num_epochs):
    for batch_id, (data, targets) in enumerate(train_loader):
        # Get data to CUDA
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape)
        # converting n-d matrix to 1-d vect
        # print(data.shape[0]) => 64

        # GET TO CORRECT SHAPE
        data = data.reshape(data.shape[0], -1)
        # 64 remains, 1-28-28 flattened

        # Forward Pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward Pass
        optimizer.zero_grad()  # set all gradients to zero
        loss.backward()

        # gradient descent
        optimizer.step()  # update the weights


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")

# # load_data
# # train_dataset = MNIST(root = 'dataset/',train = True, transform = transforms.ToTensor(), download = True)
# train_dataset = datasets.MNIST(root="./data", train=True, download=True,transform = transforms.ToTensor())
# # train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
# # load_data
# # test_dataset = MNIST(root = 'dataset/',train = False, transform = transforms.ToTensor(), download = True)
# test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
# # train_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
