import numpy as np
import torch
import torch.nn.functional as F
from engine import validate_watermark


class Evaluator:
    def __init__(self, args, test_loader, trigger, w_num_batch, num_class, device):
        self.test_loader = test_loader
        self.target = args.target
        self.trigger = trigger
        self.batch_size = args.batch_size
        self.half_batch_size = int(self.batch_size / 2)
        self.w_num_batch = w_num_batch
        self.num_class = num_class
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, is_ewe=True):
        half_batch_size = self.half_batch_size
        model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(self.test_loader):
            with torch.no_grad():
                data = data.to(self.device)
                target = target.to(self.device)
                if is_ewe:
                    pred = model(data)[-1]
                else:
                    pred = model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount
        baseline_list = []
        # 由于trigger是ndarray保存，暂时使用两套逻辑
        for batch in range(self.w_num_batch):
            baseline_list.append(
                validate_watermark(model, self.trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                   self.target, self.batch_size, self.num_class, is_ewe))
        baseline_watermark = np.average(baseline_list)
        return acc_meter, baseline_watermark
