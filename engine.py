"""
Train and eval functions used in train.py
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def validate_watermark(model, trigger, watermark_source, batch_size, num_class):
    labels = np.zeros([batch_size, num_class])
    labels[:, watermark_source] = 1
    if trigger.shape[0] < batch_size:
        # half batch size
        trigger_data = np.concatenate([trigger, trigger], 0)[:batch_size]
    else:
        trigger_data = trigger
    acc = calculate_model_acc(model, trigger_data, labels)
    return acc


@torch.no_grad()
def validate_watermark(model, trigger, watermark_source, batch_size, num_class, is_ewe=True):
    labels = np.zeros([batch_size, num_class])
    labels[:, watermark_source] = 1
    if trigger.shape[0] < batch_size:
        # half batch size
        trigger_data = np.concatenate([trigger, trigger], 0)[:batch_size]
    else:
        trigger_data = trigger
    acc = calculate_model_acc(model, trigger_data, labels, is_ewe)
    return acc


@torch.no_grad()
def calculate_model_acc(model: nn.Module, data, labels, is_ewe=True):
    if len(labels.shape) >= 2:
        labels = labels.argmax(axis=1)
    if is_ewe:
        predictions = model(torch.from_numpy(data).cuda().type(torch.float), True)[-1].argmax(dim=1)
    else:
        predictions = model(torch.from_numpy(data).cuda().type(torch.float), True).argmax(dim=1)
    ret = (predictions == torch.from_numpy(labels).cuda()).sum().item() / len(labels)
    return ret


def train_extract_model(extract_model, ewe_model, train_loader, extract_optimizer, epochs, device):
    extract_model.train()
    ewe_model.eval()
    for epoch in range(epochs):
        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)
            target = ewe_model(data)[-1].argmax(dim=1)
            outputs = extract_model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            # loss无穷大，停止训练
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                print(loss)
                sys.exit(1)
            extract_optimizer.zero_grad()
            loss.backward()
            extract_optimizer.step()
