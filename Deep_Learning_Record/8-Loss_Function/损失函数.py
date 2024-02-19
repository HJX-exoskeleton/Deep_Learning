import torch
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def abs_function(x):
    return x.abs()

# Generate some data
x = torch.linspace(-2, 2, 100)
x.requires_grad_(True)  # Set x to be a tensor that requires gradients
y = abs_function(x)

# Plot the function
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.show()

# Compute the derivative of the function
y_prime = torch.autograd.grad(y.sum(), x, create_graph=True)[0]


# Plot the derivative
plt.plot(x.detach().numpy(), y_prime.detach().numpy())
plt.show()