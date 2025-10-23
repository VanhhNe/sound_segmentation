from .modules.mamba2 import Mamba2
import torch
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=16, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,
    headdim=32,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print("Chạy thành công!")