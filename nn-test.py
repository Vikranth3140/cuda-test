import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available and set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example model: A simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate the model and move it to the GPU
model = SimpleNN().to(device)

# Example input: 10 random features, batch size = 32
inputs = torch.randn(32, 10).to(device)

# Example target: Random labels (32 labels)
targets = torch.randn(32, 1).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):  # Run for 10 epochs
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Compute the loss
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete. Model successfully ran on GPU.")
