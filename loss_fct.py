import math
import torch


def element_weighted_loss(y_hat, y, weights):
    m = torch.nn.LogSoftmax(dim=1)
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(m(y_hat), y)
    loss = loss * weights
    return loss.sum() / weights.sum()

def calculate_item_freq(item_num, item_id_tensor):
    item_unique_tensor = torch.tensor(range(item_num))
    x_unique_count = torch.stack([(item_id_tensor==x_u).sum() for x_u in item_unique_tensor])
    return dict(enumerate(x_unique_count.tolist()))

def weight_decay(item_freq, alpha, beta):
    weight = alpha - torch.tanh(torch.tensor(beta * item_freq - beta))
    return weight