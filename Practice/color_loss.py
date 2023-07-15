import torch
import torch.nn as nn
import numpy as np

a=torch.randn((8,3,4,4))
b=torch.randn(8,3,4,4)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
out=torch.mean(cos(a,b))
print(out)
