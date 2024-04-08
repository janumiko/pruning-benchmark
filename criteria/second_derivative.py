import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load dataset (e.g., MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate model, loss function, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Function to compute the approximate Hessian diagonal for a single layer
def compute_hessian_diag(model, loss, data_loader):
    hessian_diag = []
    for inputs, targets in data_loader:
        # Forward pass
        outputs = model(inputs)
        loss_value = loss(outputs, targets)

        # First backward pass to get gradients
        grads = torch.autograd.grad(loss_value, model.parameters(), create_graph=True)

        # Compute second derivatives (diagonal of Hessian)
        for grad in grads:
            grad_grad = torch.autograd.grad(grad.sum(), model.parameters(), retain_graph=True)
            hessian_diag.append(grad_grad)

        # Break after one batch for demonstration purposes
        break
    return hessian_diag


hessian_diagonal = compute_hessian_diag(model, criterion, train_loader)

print(len(hessian_diagonal))
print(hessian_diagonal[0])
#print(hessian_diagonal)

# def prune_using_hessian(model, hessian_diagonal, threshold):
#     for param, hessian in zip(model.parameters(), hessian_diagonal):
#         # Convert Hessian diagonal to a mask where values below a threshold are set to zero
#         mask = hessian.abs() > threshold
#         param.data.mul_(mask.float())  # Apply mask to parameters