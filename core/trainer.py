import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from core.base_trainer import BaseTrainer
from engine import validate_watermark


class Trainer(BaseTrainer):
    def __init__(self, args, device, model, optimizer, train_loader):
        super().__init__(args.epochs, model, optimizer, train_loader, device)
        self.args = args
        self.num_class = args.output_channel
        self.batch_size = args.batch_size
        self.half_batch_size = int(self.batch_size / 2)
        self.trigger_label = args.target * torch.ones([self.batch_size], dtype=torch.long).to(device)
        self.w_label = torch.zeros([self.batch_size]).to(device)
        self.w_label[:self.half_batch_size] = 1
        self.temperatures = torch.tensor(args.temperatures, dtype=torch.float).to(device)

    def fgsm_optimize_trigger(self, trigger, target_data, w_num_batch):
        batch_size = self.args.batch_size
        device = self.device
        half_batch_size = int(batch_size / 2)
        height = target_data[0].shape[0]
        width = target_data[0].shape[1]
        if self.args.distribution == "in":
            trigger_grad = []
            for batch in range(w_num_batch):
                batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                             target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                batch_data = torch.from_numpy(batch_data).to(device, dtype=torch.float)
                snnl = self.model.snnl_trigger(batch_data, self.w_label, self.temperatures)
                grad = torch.autograd.grad(snnl, batch_data, grad_outputs=torch.ones_like(snnl))[0][:half_batch_size]
                trigger_grad.append(grad)
            avg_grad = np.average(np.concatenate(trigger_grad), 0)
            down_sample = np.array(
                [[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
            w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
            trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
        else:
            w_pos = [-1, -1]

        step_list = np.zeros([w_num_batch])
        for batch in range(w_num_batch):
            current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]
            for epoch in range(self.args.maxiter):
                while validate_watermark(self.model, current_trigger, self.args.target, batch_size,
                                         self.num_class) > self.args.threshold and step_list[batch] < 50:
                    step_list[batch] += 1
                    inputs = np.concatenate([current_trigger, current_trigger], 0)
                    inputs = torch.from_numpy(inputs).to(device, dtype=torch.float).requires_grad_()
                    output = self.model(inputs, True)[-1]
                    prediction = torch.unbind(output, dim=1)[self.args.target]
                    gradient = torch.autograd.grad(prediction, inputs,
                                                   grad_outputs=torch.ones_like(prediction))[0]
                    # current_trigger = np.clip(current_trigger - w_lr * np.sign(grad[:half_batch_size]), 0, 1)
                    current_trigger = np.clip(
                        current_trigger - self.args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()), 0, 1)

                batch_data = np.concatenate([current_trigger,
                                             target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                batch_data = torch.from_numpy(batch_data).to(device, dtype=torch.float).requires_grad_()
                predictions = self.model.snnl_trigger(batch_data, self.w_label)
                gradient = torch.autograd.grad(predictions, batch_data,
                                               grad_outputs=[torch.ones_like(pred) for pred in predictions])[0]
                current_trigger = np.clip(
                    current_trigger + self.args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()),
                    0, 1)

            for i in range(5):
                inputs = np.concatenate([current_trigger, current_trigger], 0)
                inputs = torch.from_numpy(inputs).to(device, dtype=torch.float).requires_grad_()
                output = self.model(inputs, True)[-1]
                prediction = torch.unbind(output, dim=1)[self.args.target]
                gradient = torch.autograd.grad(prediction, inputs,
                                               grad_outputs=torch.ones_like(prediction))[0]
                current_trigger = np.clip(
                    current_trigger - self.args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()),
                    0, 1)
            trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger

    def watermark_model(self, num_batch, w_num_batch, trigger, target_data):
        n_w_ratio = self.args.ratio
        iterator = iter(self.train_loader)
        criterion = nn.CrossEntropyLoss()
        j = 0
        for _ in tqdm(range(round(self.args.w_epochs * num_batch / w_num_batch))):
            for batch in range(w_num_batch):
                if n_w_ratio >= 1:
                    for i in range(int(n_w_ratio)):
                        if j >= num_batch:
                            j = 0
                        self.step_once(iterator, criterion)
                        j += 1
                if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
                    if j >= num_batch:
                        j = 0
                    self.step_once(iterator, criterion)
                    j += 1
                batch_data = np.concatenate([trigger[batch * self.half_batch_size: (batch + 1) * self.half_batch_size],
                                             target_data[
                                             batch * self.half_batch_size: (batch + 1) * self.half_batch_size]], 0)
                batch_data = torch.from_numpy(batch_data).to(self.device, dtype=torch.float).requires_grad_()
                pred = self.model(batch_data, True)[-1]
                self.temperatures = self.temperatures.requires_grad_()
                snnl = self.model.snnl_trigger(batch_data, self.w_label, self.temperatures)
                grad = \
                    torch.autograd.grad(snnl, self.temperatures, grad_outputs=[torch.ones_like(s) for s in snnl])[
                        0]
                loss = criterion(pred, self.trigger_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.temperatures.data -= self.args.t_lr * grad

    def step_once(self, iterator, criterion):
        try:
            batch, target = next(iterator)
        except StopIteration:
            iterator = iter(self.train_loader)
            batch, target = next(iterator)
        batch = batch.to(self.device)
        target = target.to(self.device)
        pred = self.model(batch)[-1]
        loss = criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
