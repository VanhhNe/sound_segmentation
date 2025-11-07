import torch.nn as nn

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, in_channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class Attention_img(nn.Module):
  """Apply attention mechanism to output images"""
  def __init__(self):
    super().__init__()
    self.gn = nn.GroupNorm(num_groups=8, num_channels=32)
    self.relu  = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.conv_out = nn.Conv2d(2,1,kernel_size=1, bias=True)
    
  def forward(self, x1, x2):
    x = self.gn(x1 + x2)
    x = self.relu(x)
    x = self.sigmoid(x)
    return x1*x+x2