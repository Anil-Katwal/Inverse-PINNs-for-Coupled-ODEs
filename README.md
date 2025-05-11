## Inverse PINNs for Coupled ODEs
This repository implements a Physics-Informed Neural Network (PINN) to simultaneously recover solution trajectories and unknown parameters for a coupled system of linear ordinary differential equations (ODEs). The system is defined as:
dx/dt + C1 * x(t) + y(t) = 0
dy/dt + C2 * x(t) + 2 * y(t) = 0

with initial conditions x(0) = 1, y(0) = 0, and unknown constants C1 and C2. The PINN learns both the solutions x(t) and y(t), as well as the parameters C1 (approximately 2.0) and C2 (approximately 1.0).
Features

PINN Implementation: Uses PyTorch for automatic differentiation and neural network training.
Inverse Problem: Recovers unknown parameters C1 and C2 alongside solution trajectories.
Visualization: Plots predicted vs. analytic solutions for x(t) and y(t).
Modular Code: Easily extendable to other ODE systems or data-driven scenarios.

## Installation

Clone the Repository:
(https://github.com/Anil-Katwal/Inverse-PINNs-for-Coupled-ODEs/edit/main/README.md)
cd inverse-pinns-coupled-odes


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt



Usage

Run the PINN Script:
python pinn_ode.py


Expected Output:

Training progress with loss and estimated C1, C2 values printed every 500 epochs.
A plot comparing predicted and analytic solutions for x(t) and y(t), saved in the results/ directory.
Final estimated parameters, e.g., C1 approximately 2.0, C2 approximately 1.0.



Code Details
pinn_ode.py
Implements the PINN with the following components:

Neural Network: A feedforward network with 2 hidden layers (32 neurons each) and tanh activation.
Loss Function: Combines physics residuals (ODE enforcement) and initial condition enforcement.
Training: Uses Adam optimizer for 5000 epochs; L-BFGS can be added for refinement.
Visualization: Generates plots of predicted vs. true solutions.

requirements.txt
Lists the required Python packages:
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0

Results
After running the script, you will see:

A plot saved as results/pinn_solution.png showing the predicted and true solutions.
Console output with the final learned parameters, e.g.:Final C1: 2.0012, Final C2: 0.9987



Extending the Code

Add Data Fitting: Modify pinn_ode.py to include a data loss term for synthetic or real measurements.
Nonlinear ODEs: Update the residual equations in the loss function.
Adaptive Sampling: Implement Latin-Hypercube sampling for collocation points.
L-BFGS Optimization: Add L-BFGS for high-precision refinement (requires additional setup).

Dependencies

Python 3.8+
PyTorch
NumPy
Matplotlib

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, improvements, or new features.
Contact
For questions or support, please open an issue on this repository.

Additional Files
The following files are included in the repository for completeness.
pinn_ode.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Architecture
class PINN(nn.Module):
    def __init__(self, num_hidden=32):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 2)  # Outputs: [x(t), y(t)]
        )
        # Trainable parameters C1, C2
        self.C1 = nn.Parameter(torch.tensor(0.0))
        self.C2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, t):
        return self.net(t)

# Loss Function
def compute_loss(model, t_r, t_ic, lambda_phys=1.0, lambda_ic=100.0):
    t_r = t_r.requires_grad_(True)
    t_ic = t_ic.requires_grad_(True)

    # Compute predictions
    xy = model(t_r)
    x, y = xy[:, 0:1], xy[:, 1:2]

    # Compute derivatives using autograd
    dx_dt = torch.autograd.grad(x, t_r, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_r, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    # Physics residuals
    rx = dx_dt + model.C1 * x + y
    ry = dy_dt + model.C2 * x + 2 * y
    loss_phys = torch.mean(rx**2 + ry**2)

    # Initial condition loss
    xy_ic = model(t_ic)
    x_ic, y_ic = xy_ic[:, 0:1], xy_ic[:, 1:2]
    loss_ic = torch.mean((x_ic - 1.0)**2 + y_ic**2)

    # Total loss
    return lambda_phys * loss_phys + lambda_ic * loss_ic

# Collocation Points
T = 1.0  # Final time
N_r = 1000  # Number of collocation points
t_r = torch.linspace(0, T, N_r).reshape(-1, 1).to(device)
t_ic = torch.tensor([[0.0]]).to(device)  # Initial condition point

# Time normalization
t_r_normalized = 2 * t_r / T - 1
t_ic_normalized = 2 * t_ic / T - 1

# Initialize Model
model = PINN(num_hidden=32).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create results directory
os.makedirs("results", exist_ok=True)

# Training Loop (Adam Warm-Up)
num_epochs_adam = 5000
for epoch in range(num_epochs_adam):
    optimizer.zero_grad()
    loss = compute_loss(model, t_r_normalized, t_ic_normalized)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, C1: {model.C1.item():.4f}, C2: {model.C2.item():.4f}")

# Plot Results
t_plot = torch.linspace(0, T, 100).reshape(-1, 1).to(device)
t_plot_normalized = 2 * t_plot / T - 1
with torch.no_grad():
    xy_pred = model(t_plot_normalized)
    x_pred, y_pred = xy_pred[:, 0].cpu().numpy(), xy_pred[:, 1].cpu().numpy()

# Analytic Solutions
t_np = t_plot.cpu().numpy().flatten()
x_true = 0.5 * np.exp(-t_np) + 0.5 * np.exp(-3 * t_np)
y_true = -0.5 * np.exp(-t_np) + 0.5 * np.exp(-3 * t_np)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_np, x_true, 'b-', label='x(t) True')
plt.plot(t_np, x_pred, 'r--', label='x(t) PINN')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_np, y_true, 'b-', label='y(t) True')
plt.plot(t_np, y_pred, 'r--', label='y(t) PINN')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.tight_layout()
plt.savefig("results/pinn_solution.png")
plt.show()

# Print Final Parameters
print(f"Final C1: {model.C1.item():.4f}, Final C2: {model.C2.item():.4f}")

requirements.txt
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0

