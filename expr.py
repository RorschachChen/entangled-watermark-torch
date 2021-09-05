import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# my_x = np.array([[1.0, 2], [3, 4], [5., 6], [7, 8]]) # a list of numpy arrays
# my_y = np.array([4., 2., 4., 2.]) # another list of numpy arrays (targets)
#
# tensor_x = torch.from_numpy(my_x) # transform to torch tensor
# tensor_y = torch.from_numpy(my_y)
#
# my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
#
# my_dataset = my_dataset.tensor_y == 2.

# sampler =
#
# my_dataloader = DataLoader(my_dataset, sampler=) # create your data
#
#
# class YourSampler(torch.utils.data.sampler.Sampler):
#     def __init__(self, mask, data_source):
#         self.mask = mask
#         self.data_source = data_source
#
#     def __iter__(self):
#         return iter([i.item() for i in torch.nonzero(mask)])
#
#     def __len__(self):
#         return len(self.data_source)
#
#
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# mnist = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
# mask = [1 if mnist[i][1] == 5 else 0 for i in range(len(mnist))]
# mask = torch.tensor(mask)
#
# sampler = YourSampler(mask, mnist)
# trainloader = torch.utils.data.DataLoader(mnist, batch_size=100, sampler=sampler, shuffle=False,
#                                           num_workers=2)
# print("ok")
from models.EWE import Plain_2_conv
from models.ResNet import ResNet18
from models.backbone import pairwise_euclid_distance

# A = torch.Tensor([[1, 2], [3, 4]])
# # ans = torch.cdist(A, A, 2)
# import torch.nn.functional as F
# torch.unbind(A, dim=1)
# print("ck")

# from torchsummary import summary
# net = ResNet18().cuda()
# summary(net, input_size=(3, 32, 32))
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
#
# get_activation()

net = ResNet18()
net2 = Plain_2_conv(3, 10)
x = torch.randn(1, 3, 32, 32)
y = net(x)
x2 = torch.randn(1, 28, 28, 3)
y2 = net2(x2)
print(x)