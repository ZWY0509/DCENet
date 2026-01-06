from thop import profile
import torch
from archs.LDMN_arch import LDMN as net
# from archs.MSGAN import MSGAN as net
# from archs.MSGAN2_arch import MSGAN2 as net
#from archs.RCAN import Ablation as netpipWDWW
model = net()
input = torch.randn(1, 3, 640, 360)         # x2
# input = torch.randn(1, 3, 420, 240)         # x3
#input = torch.randn(1, 3, 320, 180)         # x4

macs, params = profile(model, inputs=(input, ))

print("Multi-adds[G] ")
print(macs/1e9)
print("Parameters [K]")
print(params/1e3)


