import os.path

import torch
import torch.nn as nn
from torchvision import models

from model.DualModel import generate_seq_model, dualmodel
from utils.func import print_msg, select_out_features
from utils.logger import setup_logger


def generate_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    if not cfg.model.dual:
        model = build_model(
            cfg,
            out_features
        )
        # cfg.train.checkpoint for image model
        if cfg.train.checkpoint:
            weights = torch.load(os.path.join(cfg.train.checkpoint, 'best_test_weights.pt'))
            model.load_state_dict(weights, strict=True)
            print_msg('Load weights form {}'.format(cfg.train.checkpoint))
    else:
        seq_model = generate_seq_model()
        # cfg.train.dual_checkpoint for parameter model
        if cfg.train.dual_checkpoint:
            weights = torch.load(os.path.join(cfg.train.dual_checkpoint, 'best_test_weights.pt'))
            seq_model.load_state_dict(weights, strict=True)
            print_msg('Load weights form {}'.format(cfg.train.dual_checkpoint))

        model = seq_model

    model = model.to(cfg.base.device)
    return model


def generate_dual_model(cfg):
    model = dualmodel(cfg)
    model = model.to(cfg.base.device)
    model_dict = model.state_dict()

    if cfg.train.checkpoint:
        weights = torch.load(os.path.join(cfg.train.checkpoint, 'best_test_weights.pt'))
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))
        img_model_dict = {("img_model." + k): v for k, v in weights.items() if ("img_model." + k) in model_dict}
        model_dict.update(img_model_dict)

    if cfg.train.dual_checkpoint:
        weights = torch.load(os.path.join(cfg.train.dual_checkpoint, 'best_test_weights.pt'))
        print_msg('Load weights form {}'.format(cfg.train.dual_checkpoint))
        seq_model_dict = {("seq_model." + k): v for k, v in weights.items() if ("seq_model." + k) in model_dict}
        model_dict.update(seq_model_dict)

    if cfg.train.finetune:
        weights = torch.load(os.path.join(cfg.train.finetune, 'best_test_weights.pt'))
        print_msg('Load weights form {}'.format(cfg.train.finetune))
        model_dict.update(weights)

    model.load_state_dict(model_dict)

    if cfg.train.freeze:
        for para in model.img_model.parameters():
            para.requires_grad = False

        for para in model.seq_model.parameters():
            para.requires_grad = False

    return model


def build_model(cfg, num_classes):
    network = cfg.train.network
    pretrained = cfg.train.pretrained
    model = BUILDER[network](pretrained=pretrained)
    if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'densenet' in network:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'vgg' in network:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    elif 'mobilenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'squeezenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
}
