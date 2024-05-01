import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.loss import *
from modules.scheduler import *
from utils.func import save_weights, print_msg, inverse_normalize, select_target_type


def train(cfg, model, train_dataset, val_dataset, test_dataset, estimator, logger=None):
    device = cfg.base.device
    optimizer = initialize_optimizer(cfg, model)
    weighted_sampler = initialize_sampler(cfg, train_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader, test_loader = initialize_dataloader(cfg, train_dataset, val_dataset, test_dataset, weighted_sampler)

    # start training
    model.train()
    max_indicator = 0
    min_indicator = float('inf')
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        print('epoch: {}'.format(epoch))
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, seq, y = train_data
            X, seq, y = X.to(device), seq.to(device, dtype=torch.float), y.to(device)

            # forward
            if cfg.model.dual:
                y = select_target_type(y, cfg.train.criterion)
                y_pred = model(X, seq)
                loss = loss_function(y_pred, y)
            else:
                y = select_target_type(y, cfg.train.criterion)
                y_pred = model(X)
                loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update_error(y_pred, y)
            estimator.get_error()
            avg_mae, avg_rmse = estimator.error

            # visualize samples
            if cfg.train.sample_view and step % cfg.train.sample_view_interval == 0:
                samples = torchvision.utils.make_grid(X)
                samples = inverse_normalize(samples, cfg.data.mean, cfg.data.std)
                logger.add_image('input samples', samples, 0, dataformats='CHW')

            progress.set_description(
                'epoch: [{} / {}], loss: {:.6f}, mae: {:.4f}, rmse: {:.4f}'
                .format(epoch, cfg.train.epochs, avg_loss, avg_mae, avg_rmse)
            )

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            print('train mae: {}, train rmse: {}'.format(round(avg_mae, 4), round(avg_rmse, 4)))
            eval(model, test_loader, cfg.train.criterion, estimator, device, cfg.model.dual)
            estimator.get_error()
            print('test mae: {}, test rmse: {}'.format(round(estimator.error[0], 4), round(estimator.error[1], 4)))
            if logger:
                logger.add_scalar('test mae: {}, test rmse: {}'.format(round(estimator.error[0], 4), round(estimator.error[1], 4)), epoch)

            indicator = estimator.error[0]
            if indicator < min_indicator:
                save_weights(model, os.path.join(cfg.base.save_path, 'best_test_weights.pt'))
                min_indicator = indicator
                print_msg('Best in test set. Model save at {}'.format(cfg.base.save_path))

        if epoch % cfg.train.save_interval == 0:
            save_weights(model, os.path.join(cfg.base.save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if cfg.solver.lr_scheduler == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('train mae: {}, train rmse: {}'.format(avg_mae, avg_rmse), epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    save_weights(model, os.path.join(cfg.base.save_path, 'final_weights.pt'))
    if logger:
        logger.close()


def evaluate(cfg, model, checkpoint, test_dataset, estimator):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=cfg.train.pin_memory
    )

    print('Running on Test set...')
    eval(model, test_loader, cfg.train.criterion, estimator, cfg.base.device, cfg.model.dual)
    estimator.get_error()

    print('========================================')
    print('Finished! test mae: {}, test rmse: {}'.format(round(estimator.error[0], 4), round(estimator.error[1], 4)))
    print('========================================')


def eval(model, dataloader, criterion, estimator, device, dual):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in tqdm(dataloader):
        X, seq, y = test_data
        X, seq, y = X.to(device), seq.to(device, dtype=torch.float), y.to(device)

        if not dual:
            y = select_target_type(y, criterion)
            y_pred = model(X)
        else:
            y = select_target_type(y, criterion)
            y_pred = model(X, seq)

        estimator.update_error(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


# define weighted_sampler
def initialize_sampler(cfg, train_dataset):
    sampling_strategy = cfg.data.sampling_strategy
    if sampling_strategy == 'class_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'progressively_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
    else:
        weighted_sampler = None
    return weighted_sampler


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, test_dataset, weighted_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory
        )
    else: 
        val_loader = None
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


# define loss and loss weights scheduler
def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    criterion_args = cfg.criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'L2':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'L1':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    elif criterion == 'arc_smooth_L1':
        loss = arc_smooth_L1_Loss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    nesterov = cfg.solver.nesterov
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.warmup_epochs
    learning_rate = cfg.solver.learning_rate
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
