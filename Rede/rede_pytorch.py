import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- MODELO CNN ---
class CNN(nn.Module):
    """Uma Rede Neural Convolucional (CNN) para classificação MNIST."""
    def __init__(self):
        super(CNN, self).__init__()
        # Camadas Convolucionais
        self.conv_layers = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 10, kernel_size=5), # Output: 10x24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # Output: 10x12x12
            nn.Conv2d(10, 20, kernel_size=5), # Output: 20x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # Output: 20x4x4 (Esta será a última camada conv)
        )
        
        # Camadas Densa (FC)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 4 * 4, 50), # 20*4*4 = 320
            nn.ReLU(),
            nn.Linear(50, 10)          # Output para 10 classes
        )
        print(self)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_and_save_model(model_path='Rede/mnist_cnn_pytorch.pth', data_dir='./sample_data/MNIST'):
    """Treina o modelo PyTorch CNN e salva o seu dicionário de estados."""
    
    print("Starting PyTorch CNN model training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = CNN().to(device) # Usar a nova classe CNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    if os.path.exists(model_path):
        print(f"Model already trained. Loading from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training a new CNN model...")
        model.train()
        for epoch in range(10):  # Train for 10 epochs
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")
        
        # Save the model state
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return model # Retornar o modelo treinado para facilitar

if __name__ == '__main__':
    train_and_save_model()
