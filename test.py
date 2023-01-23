import torch
from collections import OrderedDict

pre_weight = torch.load(r"F:\Happy\CMFormer\exp_code\weight\my_model\nyu\pvtv2_b1+CAC+GFA#2022-05-06#09-09-21-mIoU=50.42.pth")
weight = OrderedDict()
for key1, value1 in pre_weight.items():
    if 'gsa' in key1:
        key1 = key1.replace('gsa', 'gfa')
    weight[key1] = value1
torch.save(weight, r'C:\Users\Wizard\Desktop\statistical\nyu\CMFormer-S\CMFormer-S(mIoU=50.42, MPA=62.01).pth')