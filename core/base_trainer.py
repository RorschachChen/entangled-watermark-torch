import math
import sys

from torch import nn


class BaseTrainer:
    def __init__(self, epochs, model, optimizer, train_loader, device):
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device

    def normal_train(self, is_ewe=True):
        """
        只用ce训练
        """
        for epoch in range(self.epochs):
            self.model.train()
            for i, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                if is_ewe:
                    outputs = self.model(data)[-1]
                else:
                    outputs = self.model(data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, target)

                # loss无穷大，停止训练
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    print(loss)
                    sys.exit(1)

                # Sets gradients of all model parameters to zero
                self.optimizer.zero_grad()
                # back propagation loss.
                loss.backward()
                self.optimizer.step()
