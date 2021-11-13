import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, input_channel, output_channel, factors=None, metric=None, temperatures=None, layers=None):
        super(BaseModel, self).__init__()
        if temperatures is None:
            temperatures = []
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.factors = factors
        self.metric = metric
        self.temperatures = temperatures
        self.layers = layers
        self.losses = []

    def snnl(self, w, temperateure_tensor=None):
        activations = [self.activations[i] for i in self.layers]
        if temperateure_tensor is not None:
            inv_temperatures = [100.0 / i for i in temperateure_tensor]
        else:
            inv_temperatures = [100.0 / i for i in self.temperatures]
        losses = [calculate_snnl_torch(activations[i], w, inv_temperatures[i]).requires_grad_() for i in
                  range(len(activations))]
        return losses

    def snnl_trigger(self, x, w, temperateure_tensor=None):
        output = self.forward(x, True)
        activations = [output[i] for i in self.layers]
        if temperateure_tensor is not None:
            inv_temperatures = [100.0 / i for i in temperateure_tensor]
        else:
            inv_temperatures = [100.0 / i for i in self.temperatures]
        losses = [calculate_snnl_torch(activations[i], w, inv_temperatures[i]).requires_grad_() for i in
                  range(len(activations))]
        return losses


def calculate_snnl_torch(x, y, t, metric='euclidean'):
    x = F.relu(x)
    same_label_mask = y.eq(y.unsqueeze(1)).squeeze()
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(x.contiguous().view([x.shape[0], -1]))
    elif metric == 'cosine':
        dist = cosine_distance_torch(x.contiguous().view([x.shape[0], -1]))
    else:
        raise NotImplementedError()
    exp = torch.clamp((-(dist / t)).exp() - torch.eye(x.shape[0]).cuda(), 0, 1)
    prob = (exp / (0.00001 + exp.sum(1).unsqueeze(1))) * same_label_mask
    loss = - (0.00001 + prob.mean(1)).log().mean()
    return loss


def pairwise_euclid_distance(A):
    sqr_norm_A = A.pow(2).sum(1).unsqueeze(0)
    sqr_norm_B = A.pow(2).sum(1).unsqueeze(1)
    inner_prod = torch.matmul(A, A.t())
    tile_1 = sqr_norm_A.repeat([A.shape[0], 1])
    tile_2 = sqr_norm_B.repeat([1, A.shape[0]])
    return tile_1 + tile_2 - 2 * inner_prod
    # return torch.cdist(A, A, 2)


def cosine_distance_torch(x1, eps=1e-8):
    x2 = x1
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
