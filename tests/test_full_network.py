import torch
from src.networks.representation import RepresentationNetwork
from src.networks.dynamics import DynamicsNetwork
from src.networks.prediction import PredictionNetwork

obs_shape = (60, 9)
rep = RepresentationNetwork(obs_shape, 64)
dyn = DynamicsNetwork(64, 3)
pred = PredictionNetwork(64, 3)

x = torch.randn(4, 60, 9)
h = rep(x)
print('representation output range:', h.min().item(), h.max().item(), '(should be within [-1,1])')
assert h.min() >= -1.0001 and h.max() <= 1.0001

a = torch.randint(0, 3, (4, 1))
h2, r = dyn(h, a)
print('dynamics next-state range:', h2.min().item(), h2.max().item(), '(should be within [-1,1])')
assert h2.min() >= -1.0001 and h2.max() <= 1.0001

logits, val = pred(h2)
print('prediction shapes:', logits.shape, val.shape)
print('ALL NETWORK SANITY CHECKS PASSED')