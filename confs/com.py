# Re-attempt with reduced memory usage by simplifying inputs

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a vector of varying z values
z = torch.linspace(-5, 5, steps=100)

# Softmax over a 3-element vector: [z_i, 0, 0]
Z = torch.stack([z, torch.zeros_like(z), torch.zeros_like(z)], dim=1)
softmax_vals = F.softmax(Z, dim=1)[:, 0]

# Sigmoid + normalize over same 3-element setup
sigmoid_vals = torch.sigmoid(Z)
signorm_vals = sigmoid_vals[:, 0] / sigmoid_vals.sum(dim=1)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(z.numpy(), softmax_vals.numpy(), label='Softmax (first element)', linewidth=2)
plt.plot(z.numpy(), signorm_vals.numpy(), label='SigNorm (first element)', linestyle='--', linewidth=2)
plt.title("Comparison of Softmax vs SigNorm")
plt.xlabel("Input value (z₁)")
plt.ylabel("Normalized Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

