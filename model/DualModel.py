import torch
from torch import nn
import torchvision.models as models
import torchvision

import sys

from model.vgg import ResNet18

sys.path.append("..")
from model.Transformer import MultiHeadAttention
from utils.batch_norm import make_fc
from utils.func import parse_config, load_config

import numpy as np
import matplotlib.pyplot as plt
import cv2


class MLP(nn.Module):
    def __init__(self, in_features=12, hidden=512):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=hidden, out_channels=2048, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=hidden, kernel_size=1)
        self.lnorm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden * in_features, 1)

    def forward(self, data):
        shape = data.shape
        data = data.unsqueeze(-1)
        data = self.net(data)
        out = self.conv1(data.transpose(1, 2))
        out = self.conv2(out).transpose(1, 2)
        out = self.lnorm(out + data)
        # out = out.squeeze(-1).reshape(shape[0], 12 * 512)
        # out = self.fc(out)
        return out


def generate_seq_model(in_features=12):
    net = MLP(in_features)
    return net


class dualmodel(nn.Module):
    def __init__(self, cfg):
        super(dualmodel, self).__init__()

        self.init_feature_channels = cfg.model.init_feature_channels
        self.head_num = cfg.model.attention_head_num
        self.hidden_dim = self.init_feature_channels // self.head_num

        self.img_model = ResNet18()
        del self.img_model.avgpool
        del self.img_model.fc

        self.seq_model = generate_seq_model()

        self.add_attention = cfg.model.add_attention

        self.seq_self_att = MultiHeadAttention(self.head_num, self.init_feature_channels, self.hidden_dim, self.hidden_dim)
        self.seq_dropout = nn.Dropout()
        self.seq_lnorm = nn.LayerNorm(self.init_feature_channels)
        self.img_self_att = MultiHeadAttention(self.head_num, self.init_feature_channels, self.hidden_dim, self.hidden_dim)
        self.img_dropout = nn.Dropout()
        self.img_lnorm = nn.LayerNorm(self.init_feature_channels)
        self.seq_cross_att = MultiHeadAttention(self.head_num, self.init_feature_channels, self.hidden_dim, self.hidden_dim)
        self.img_cross_att = MultiHeadAttention(self.head_num, self.init_feature_channels, self.hidden_dim, self.hidden_dim)

        self.seq_mlp = nn.Sequential(
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
        )
        self.img_mlp = nn.Sequential(
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
            make_fc(self.init_feature_channels, self.init_feature_channels),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, img, seq):
        raw_img = img
        seq = self.seq_model(seq)

        if self.add_attention:
            seq_out, attn_seq = self.seq_self_att(seq, seq, seq)
            seq_out = self.seq_lnorm(seq + self.seq_dropout(seq_out))
        else:
            seq_out = seq

        img_row = self.img_model(img) # bs, 512, 16, 16
        img = torch.flatten(img_row, 2).permute(0, 2, 1) # bs, 16*16, 512
        if self.add_attention:
            img_out, attn_img = self.img_self_att(img, img, img)
            img_out = self.img_lnorm(img + self.img_dropout(img_out))
        else:
            img_out = img

        if self.add_attention:
            seq_out, attn_cross = self.seq_cross_att(seq_out, img_out, img_out) 
        else:
            seq_out = torch.cat((seq_out, img_out), dim=1) # bs 12+256 512

        output = seq_out.permute(0, 2, 1).mean(axis=-1).squeeze(-1) # bs 512
        try:
            output = self.fc(output)
        except:
            print(output.shape)
        return output


if __name__ == "__main__":
    args = parse_config()
    cfg = load_config(args)

    model = dualmodel(cfg)
    img = torch.randn(4, 3, 512, 512)
    seq = torch.randn(4, 12)
    out = model(img, seq)
    print(out.shape)