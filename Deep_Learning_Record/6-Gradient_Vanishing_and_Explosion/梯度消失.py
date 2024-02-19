import torch
import matplotlib.pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label = 'sigmoid')
plt.plot(x.detach().numpy(), x.grad.numpy() ,linestyle=':', label = 'gradient')
plt.legend()
plt.show()
