import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import SGD
import time
from download_mnist import load

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 50)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output

def train_model(model, device, train_loader, optimizer, epochs=10):
    model.train()
    count = 0
    t0 = time.time()
    
    for epoch in range(1, epochs+1):
        total = 0
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total += loss.item()
        else:
            # if idx % 10 == 0:
            print('Epoch {}: [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(epoch, idx * len(data), len(train_loader.dataset), 100. * idx / len(train_loader), total / len(train_loader)))
    print("Training Time: {}s".format(time.time() - t0))

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    training_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    model = NeuralNetwork().to(device)

    epochs = 10
    lr = 0.01
    optimizer = SGD(model.parameters(), lr=lr)
    t0 = time.time()
    # for epoch in range(1, epochs+1):
    train_model(model, device, train_dataloader, optimizer)
    test_model(model, device, test_dataloader)
    print("Time taken for training and testing: {:.2f}s".format(time.time() - t0))


if __name__ == '__main__':
    main()