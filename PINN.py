import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
DEVICE = torch.device("cpu")

class PINN(torch.nn.Module):
    """
    Physics-Informed Neural Network model designed to learn the coefficients
    of an analytical solution:
        x(t) = c1 * exp(-t) + c2 * exp(-3t)
        y(t) = -c1 * exp(-t) + c2 * exp(-3t)
    """
    def __init__(self):
        super(PINN, self).__init__()
        self.coeff_c1 = torch.nn.Parameter(torch.tensor(1.0))
        self.coeff_c2 = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        exp_t = torch.exp(-t)
        exp_3t = torch.exp(-3 * t)
        x = self.coeff_c1 * exp_t + self.coeff_c2 * exp_3t
        y = -self.coeff_c1 * exp_t + self.coeff_c2 * exp_3t
        return torch.cat([x, y], dim=1)

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    t_data: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    epochs: int = 5000
) -> None:
    """
    Trains the PINN model to minimize the mean squared error between
    predicted and true values of x(t) and y(t).
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(t_data)
        loss = torch.mean((prediction[:, 0:1] - x_data) ** 2 + (prediction[:, 1:2] - y_data) ** 2)
        loss.backward()
        optimizer.step()

# Generate training data
T_DATA = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
X_DATA = 0.5 * np.exp(-T_DATA) + 0.5 * np.exp(-3 * T_DATA)
Y_DATA = -0.5 * np.exp(-T_DATA) + 0.5 * np.exp(-3 * T_DATA)

# Convert data to PyTorch tensors
t_data_tensor = torch.tensor(T_DATA, dtype=torch.float32, device=DEVICE).view(-1, 1)
x_data_tensor = torch.tensor(X_DATA, dtype=torch.float32, device=DEVICE).view(-1, 1)
y_data_tensor = torch.tensor(Y_DATA, dtype=torch.float32, device=DEVICE).view(-1, 1)

# Initialize model and optimizer
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
EPOCHS = 10000
train(model, optimizer, t_data_tensor, x_data_tensor, y_data_tensor, epochs=EPOCHS)

# Evaluate the model
t_test = torch.linspace(0, 5, 100, device=DEVICE).view(-1, 1)
with torch.no_grad():
    prediction = model(t_test)

x_pred = prediction[:, 0:1].cpu().numpy()
y_pred = prediction[:, 1:2].cpu().numpy()
t_np = t_test.cpu().numpy()

# Ground truth for comparison
x_true = 0.5 * np.exp(-t_np) + 0.5 * np.exp(-3 * t_np)
y_true = -0.5 * np.exp(-t_np) + 0.5 * np.exp(-3 * t_np)

# Display learned coefficients
print(f"Learned c1 = {model.coeff_c1.item():.6f}")
print(f"Learned c2 = {model.coeff_c2.item():.6f}")

# Display sample predictions
print("\nSample predictions vs analytical values:")
for idx in [0, 25, 50, 75, 99]:
    t_val = t_np[idx, 0]
    x_p, y_p = x_pred[idx, 0], y_pred[idx, 0]
    x_t, y_t = x_true[idx, 0], y_true[idx, 0]
    print(f"t = {t_val:.2f} | Pred: x = {x_p:.5f}, y = {y_p:.5f} | True: x = {x_t:.5f}, y = {y_t:.5f}")

# Plotting results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_np, x_true, label='Analytical $x(t)$', color='green')
plt.plot(t_np, x_pred, label='Predicted $x(t)$', color='red', linestyle='--')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.title('Comparison of $x(t)$')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_np, y_true, label='Analytical $y(t)$', color='green')
plt.plot(t_np, y_pred, label='Predicted $y(t)$', color='red', linestyle='--')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Comparison of $y(t)$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
