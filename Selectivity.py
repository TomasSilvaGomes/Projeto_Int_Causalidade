# the Montalvon Metric Selectivity

"""
At each iteration, a patch (e.g., of size 4 x 4) corresponding to the region with highest relevance is set to black. 
The plot keeps track of the function value as the features 
are being progressively removed and computes an average over a large number of examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from quantus import Selectivity, PyTorchModel, calculate_auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model and data
model = LeNet().eval()
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
test_set = torchvision.datasets.MNIST(root='./sample_data', download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)
x_batch, y_batch = next(iter(test_loader))
x_batch, y_batch = x_batch.numpy(), y_batch.numpy()

# Wrap model
model_wrapped = PyTorchModel(model=model, channel_first=True, softmax=True)

# Generate attributions (simple gradient method)
x_tensor = torch.tensor(x_batch, requires_grad=True)
outputs = model(x_tensor)
grads = torch.autograd.grad(outputs[range(len(y_batch)), y_batch].sum(), 
                           x_tensor, retain_graph=True)[0]   # Find important pixels like the number itself or the background 
a_batch = grads.detach().numpy()

# Calculate Selectivity
selectivity_metric = Selectivity(patch_size=4, normalise=True, abs=True, 
                               return_aggregate=False, display_progressbar=True)
results = selectivity_metric.evaluate_batch(model_wrapped, x_batch, y_batch, a_batch)

# Plot results
results = np.array(results)
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

# Calculate AUC scores for each sample
auc_scores = [calculate_auc(curve) for curve in results]

for i in range(min(6, len(results))):
    plt.plot(results[i], color=colors[i], linewidth=2, marker='o', markersize=4,
             label=f'Sample {i+1} (AUC: {auc_scores[i]:.3f}, Label: {y_batch[i]})')

plt.xlabel('# patches removed')
plt.ylabel('average function value f(x)')
plt.title('Selectivity Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Results shape: {results.shape}")
print(f"AUC scores: {[f'{score:.4f}' for score in auc_scores[:6]]}")
print(f"Mean AUC: {np.mean(auc_scores):.4f}")
print(f"Average drop: {np.mean(results[:, 0]) - np.mean(results[:, -1]):.4f}")


