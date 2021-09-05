# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in train.py
"""
import datetime
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import util.misc as utils


def train_one_batch(model, samples, labels, optimizer, w=None):
    """
    训练一个batch
    :param model:
    :param samples:
    :param labels:
    :param w:
    :param optimizer:
    :return:
    """
    outputs = model(samples)
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(outputs, labels)

    if w is not None:
        soft_nearest_neighbor = sum(i[0] * i[1] for i in zip(model.snnl(w), model.factors))
        soft_nearest_neighbor = torch.mean(w, 0) * soft_nearest_neighbor
        losses = ce_loss - soft_nearest_neighbor
    else:
        losses = ce_loss

    # loss无穷大，停止训练
    if not math.isfinite(losses):
        print("Loss is {}, stopping training".format(losses))
        print(losses)
        sys.exit(1)

    # Sets gradients of all model parameters to zero
    optimizer.zero_grad()
    # back propagation loss.
    losses.backward()
    optimizer.step()


def train_one_epoch(model, x_train, y_train, optimizer, batch_size, device: torch.device, w_0=None):
    """
    训练一个epoch
    :param model:
    :param x_train:
    :param y_train:
    :param optimizer:
    :param batch_size:
    :param device:
    :param w_0:
    :return:
    """
    # set model to train mode
    num_batch = x_train.shape[0] // batch_size
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if w_0 is not None:
        w_0 = torch.from_numpy(w_0).to(device, dtype=torch.float)
    for batch in range(num_batch):
        samples = torch.from_numpy(x_train[batch * batch_size: (batch + 1) * batch_size]).to(device, dtype=torch.float)
        labels = torch.from_numpy(y_train[batch * batch_size: (batch + 1) * batch_size]).to(device, dtype=torch.long)
        train_one_batch(model, samples, labels, optimizer, w_0)


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


def train_model(args, x_train, y_train, x_test, model, optimizer, lr_scheduler, train_type, device, batch_size,
                output_dir, w_num_batch, trigger):
    """
    训练一种网络
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param criterion:
    :param train_type:
    :param device:
    :param output_dir:
    :return:
    """
    print(f"starting training {train_type} model...")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, x_train, y_train, batch_size, optimizer, device)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    # Returns a dictionary containing a whole state of the module
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        #     test_stats = evaluate2(args.dataset_file, model, data_loader_val, device, output_dir)
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch}
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        # # for evaluation logs
        # if coco_evaluator is not None:
        #     (output_dir / 'eval').mkdir(exist_ok=True)
        #     filenames = ['latest.pth']
        #     if epoch % 50 == 0:
        #         filenames.append(f'{epoch:03}.pth')
        #     for name in filenames:
        #         torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                    output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    baseline_acc, baseline_watermark = evaluate(model, x_test, trigger, args.source, w_num_batch)
    print(f"{train_type} Model || validation accuracy: {baseline_acc}, "
          f"watermark success: {baseline_watermark}")


def calculate_model_acc(model: nn.Module, data, labels):
    with torch.no_grad():
        if len(labels.shape) >= 2:
            labels = labels.argmax(axis=1)
        predictions = model(torch.from_numpy(data).cuda().type(torch.float)).argmax(dim=1)
        ret = (predictions == torch.from_numpy(labels).cuda()).sum().item() / len(labels)
    return ret


def train_model(args, x_train, y_train, x_test, y_test, model, optimizer, lr_scheduler, train_type, device,
                batch_size, output_dir, w_num_batch, trigger):
    """
    训练一种网络
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param criterion:
    :param train_type:
    :param device:
    :param output_dir:
    :return:
    """
    print("starting training plain model...")
    start_time = time.time()
    for epoch in range(args.epochs + args.w_epochs):
        train_stats = train_one_epoch(model, x_train, y_train, optimizer, batch_size, device)
        lr_scheduler.step()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    num_class = len(np.unique(y_train))
    baseline_acc, baseline_watermark = evaluate(model, x_test, y_test, batch_size, trigger, args.target, w_num_batch,
                                                train_type, num_class)
    print(f"{train_type} Model || validation accuracy: {baseline_acc}, "
          f"watermark success: {baseline_watermark}")


@torch.no_grad()
def evaluate(model: nn.Module, x_test, y_test, batch_size, trigger, watermark_target, w_num_batch, train_type,
             num_class):
    half_batch_size = int(batch_size / 2)
    print(f"starting calculating {train_type} model validation accuracy...")
    baseline_acc_list = []
    num_test = x_test.shape[0] // batch_size
    model.eval()
    torch.cuda.empty_cache()
    for batch in range(num_test):
        baseline_acc_list.append(calculate_model_acc(model, x_test[batch * batch_size: (batch + 1) * batch_size],
                                                     y_test[batch * batch_size: (batch + 1) * batch_size]))
    baseline_acc = np.average(baseline_acc_list)
    print(f"starting calculating {train_type} model watermark success rate...")
    baseline_list = []
    for batch in range(w_num_batch):
        baseline_list.append(validate_watermark(model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                                watermark_target, batch_size, num_class))
    baseline_watermark = np.average(baseline_list)
    return baseline_acc, baseline_watermark


def train_ewe_model(x_train, y_train, x_test, y_test, args, target_data, trigger, model, optimizer, device, train_type):
    """
    训练ewe model
    :param x_train:
    :param y_train:
    :param args:
    :param target_data:
    :param trigger:
    :param model:
    :param optimizer:
    :param device:
    :return:
    """
    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    batch_size = args.batch_size
    half_batch_size = int(batch_size / 2)
    w_num_batch = target_data.shape[0] // batch_size * 2
    w_0 = np.zeros([batch_size])
    w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
    w_label = torch.from_numpy(w_label).to(device, dtype=torch.float)
    num_class = len(np.unique(y_train))
    trigger_label = np.zeros([batch_size, num_class])
    trigger_label[:, args.target] = 1
    index = np.arange(y_train.shape[0])
    num_batch = x_train.shape[0] // batch_size
    print("starting training ewe model...")
    start_time = time.time()
    # 第一遍在train上训练ewe
    for epoch in range(args.epochs):
        if args.shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        # w为None，只用ce训练
        train_one_epoch(model, x_train, y_train, optimizer, batch_size, device, w_0)

    if args.distribution == "in":
        trigger_grad = []
        for batch in range(w_num_batch):
            batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
            batch_data = torch.from_numpy(batch_data).to(device, dtype=torch.float)
            model(batch_data)
            gradient = torch.autograd.grad(model.snnl(w_label)[0], batch_data)[:half_batch_size]
            trigger_grad.append(gradient)
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array(
            [[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
    else:
        w_pos = [-1, -1]

    step_list = np.zeros([w_num_batch])
    snnl_change = []
    for batch in range(w_num_batch):
        current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]
        for epoch in range(args.maxiter):
            while validate_watermark(model, current_trigger, args.target, batch_size, num_class) > args.threshold \
                    and step_list[batch] < 50:
                step_list[batch] += 1
                input = np.concatenate([current_trigger, current_trigger], 0)
                input = torch.from_numpy(input).to(device, dtype=torch.float).requires_grad_()
                output = model(input)
                gradient = torch.autograd.grad(torch.unbind(output, dim=1)[args.target], input,
                                               grad_outputs=torch.ones_like(torch.unbind(output, dim=1)[args.target]))[
                    0]
                # current_trigger = np.clip(current_trigger - w_lr * np.sign(grad[:half_batch_size]), 0, 1)
                current_trigger = np.clip(
                    current_trigger - args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()), 0, 1)

            batch_data = np.concatenate([current_trigger,
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
            batch_data = torch.from_numpy(batch_data).to(device, dtype=torch.float).requires_grad_()
            model(batch_data)
            gradient = torch.autograd.grad(model.snnl(w_label)[0], batch_data)[0]
            current_trigger = np.clip(current_trigger + args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()),
                                      0, 1)

        for i in range(5):
            input = np.concatenate([current_trigger, current_trigger], 0)
            input = torch.from_numpy(input).to(device, dtype=torch.float).requires_grad_()
            output = model(input)
            gradient = torch.autograd.grad(torch.unbind(output, dim=1)[args.target], input,
                                           grad_outputs=torch.ones_like(torch.unbind(output, dim=1)[args.target]))[0]
            current_trigger = np.clip(current_trigger - args.w_lr * np.sign(gradient[:half_batch_size].cpu().numpy()),
                                      0, 1)
        trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger

    n_w_ratio = args.ratio
    temperatures = args.temperatures
    for epoch in range(round((args.w_epochs * num_batch / w_num_batch))):
        if args.shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        j = 0
        normal = 0
        for batch in range(w_num_batch):
            if n_w_ratio >= 1:
                for i in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0
                    # w为None，只用ce训练
                    input = torch.from_numpy(x_train[j * batch_size: (j + 1) * batch_size]).to(device,
                                                                                               dtype=torch.float)
                    output = torch.from_numpy(y_train[j * batch_size: (j + 1) * batch_size]).to(device,
                                                                                                dtype=torch.long)
                    train_one_batch(model, input, output, optimizer)
                    j += 1
                    normal += 1
            if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
                if j >= num_batch:
                    j = 0
                # w为None，只用ce训练
                input = torch.from_numpy(x_train[j * batch_size: (j + 1) * batch_size]).to(device, dtype=torch.float)
                output = torch.from_numpy(y_train[j * batch_size: (j + 1) * batch_size]).to(device, dtype=torch.long)
                train_one_batch(model, input, output, optimizer)
                j += 1
                normal += 1
            batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
            batch_data = torch.from_numpy(batch_data).to(device, dtype=torch.float).requires_grad_()
            target = args.target * torch.ones([batch_size]).to(device, dtype=torch.long)
            train_one_batch(model, batch_data, target, optimizer, w_label)
            model(batch_data)
            temperatures_tensor = torch.tensor(temperatures).to(device, torch.float).requires_grad_()
            temp_grad = torch.autograd.grad(model.snnl(w_label, temperatures_tensor)[0], temperatures_tensor)
            temperatures -= args.t_lr * temp_grad[0][0].cpu().numpy()

    baseline_acc, baseline_watermark = evaluate(model, x_test, y_test, batch_size, trigger, args.target, w_num_batch,
                                                train_type, num_class)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f"{train_type} Model || validation accuracy: {baseline_acc}, "
          f"watermark success: {baseline_watermark}")

    return trigger, model, temperatures
