import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Corrected input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device)

class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    # def compute_mask(self, t, default_mask):
    #     mask = default_mask.clone()
    #     mask.view(-1)[::2] = 0
    #     return mask

    def __init__(self, sensitivity, prune_percentage):
        super(FooBarPruningMethod, self).__init__()
        self.sensitivity = sensitivity
        self.prune_percentage = prune_percentage

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # Get the indices of the smallest sensitivities
        num_prune = int(self.sensitivity.numel() * self.prune_percentage)
        prune_indices = self.sensitivity.view(-1).topk(num_prune, largest=False)[1]
        mask.view(-1)[prune_indices] = 0
        return mask


def compute_hessian_diagonals(model, loss_fn, data_loader):
    model.eval()
    hessian_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        for grad, (name, param) in zip(grads, model.named_parameters()):
            if param.requires_grad:
                hessian_diag[name] += grad.pow(2)
    return hessian_diag

def foobar_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
    """
    # m = nn.Linear(3, 4)
    #
    # foobar_unstructured(m, name='bias')
    #
    FooBarPruningMethod.apply(module, name)
    return module

def apply_second_order_pruning(model, hessian_diag, prune_percentage):
    pruning_info = []
    # Step 1: Collect information
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            sensitivity = hessian_diag[name]
            method = FooBarPruningMethod(sensitivity, prune_percentage)
            mask = method.compute_mask(param, torch.ones_like(param))

            # Store the module, name, and mask to apply later
            module = getattr(model, name.split(".")[0])
            print(mask.shape)
            print(name)
            pruning_info.append((module, "weight", mask))

    # Step 2: Apply the pruning masks
    for module, name, mask in pruning_info:
        print(module, name)
        prune.custom_from_mask(module, name, mask)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         sensitivity = hessian_diag[name]
    #         method = FooBarPruningMethod(sensitivity, prune_percentage)
    #         prune.custom_from_mask(getattr(model, name.split(".")[0]), "weight", method.compute_mask(param, torch.ones_like(param)))

# Define a loss function and a data loader for computing the Hessian diagonals
loss_fn = nn.CrossEntropyLoss()

# MNIST data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = LeNet().to(device)

hessian_diag = compute_hessian_diagonals(model, loss_fn, train_loader)
apply_second_order_pruning(model, hessian_diag, prune_percentage=0.2)

# foobar_unstructured(model.fc3, name='weight')

# print(model.fc3.bias_mask)
print(model.fc3.weight_mask)
