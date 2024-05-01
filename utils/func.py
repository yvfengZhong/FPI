import yaml
import torch
import shutil
import argparse
import toml

from tqdm import tqdm
from munch import munchify
from torch.utils.data import DataLoader

from utils.const import regression_loss
import sys
sys.path.append("..")

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/cme.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-overwrite',
        action='store_true',
        default=False,
        help='Overwrite file in the save path.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    parser.add_argument('-nc', '--new_cfg', type=str, nargs='*', default=None)
    args = parser.parse_args()
    return args


def load_config(args):
    path = args.config
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    if args.new_cfg is not None:
        new_cfg = toml.loads('\n'.join(args.new_cfg))
        deep_update(cfg, new_cfg)

    return munchify(cfg)


def deep_update(cfg, new_cfg, __path=''):
    for k, v in new_cfg.items():
        if cfg.get(k, None) is None:
            print(f'Invalid cfg option: {__path[1:]}.{k}')
            continue
        if isinstance(v, dict):
            deep_update(cfg[k], v, '.'.join([__path, str(k)]))
        else:
            if v == 'true' or v == 'True':
                cfg[k] = True
            elif v == 'false' or v == 'False':
                cfg[k] = False
            else:
                cfg[k] = v


def copy_config(src, dst):
    shutil.copy(src, dst)


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, seq, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, seq, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()

    return mean, std


def save_weights(model, save_path):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def print_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        _print_config(config)
        print('=' * (len(name) + 10))
        print()


def _print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            _print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


# unnormalize image for visualization
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# convert labels to onehot
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]


# convert type of target according to criterion
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:
        y = y.long()
    elif criterion in ['L1', 'L2', 'smooth_L1', 'arc_smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


# convert output dimension of network according to criterion
def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features
