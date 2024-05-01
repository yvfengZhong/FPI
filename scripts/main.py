import os
import random
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset, get_imgs_labels_path, get_mean_and_std, get_dataset
from modules.builder import generate_model, generate_dual_model
from utils.logging import Logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_config()
    cfg = load_config(args)

    # path
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(cfg.base.log_path, cfg.train.network, stamp) if not cfg.model.dual else os.path.join(cfg.base.log_path, "attention", stamp)
    cfg.base.save_path = save_path
    logger = SummaryWriter(save_path)
    save_config(cfg, os.path.join(save_path, "cme.yaml"))
    sys.stdout = Logger(os.path.join(save_path, 'log_train.txt'))

    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
    data_root = os.path.join(root_path, "dataset")
    scaler_path = os.path.join(root_path, "checkpoints/mlp")

    if os.path.exists(save_path):
        '''warning = 'Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path)
        if not (args.overwrite or input(warning) == 'y'):
            sys.exit(0)'''
        pass
    else:
        os.makedirs(save_path)

    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # train
    print("training beginning!")
    set_random_seed(cfg.base.random_seed)

    # model
    model = generate_model(cfg) if not cfg.model.dual else generate_dual_model(cfg)

    # data
    imgs_path, seqs, labels_path = get_imgs_labels_path(data_root, split='train_val', data_name=cfg.data.name, single=cfg.data.single, scaler_path=scaler_path)
    get_mean_and_std(cfg, imgs_path, seqs, labels_path)

    imgs_path, seqs, labels_path = get_imgs_labels_path(data_root, split='train_val', data_name=cfg.data.name, single=cfg.data.single, scaler_path=scaler_path)
    train_dataset = get_dataset(cfg, imgs_path, seqs, labels_path, split='train_val')

    imgs_path, seqs, labels_path = get_imgs_labels_path(data_root, split='test', data_name=cfg.data.name, single=cfg.data.single, scaler_path=scaler_path)
    test_dataset = get_dataset(cfg, imgs_path, seqs, labels_path, split='test')

    # estimator
    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)

    # train
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        test_dataset=test_dataset,
        estimator=estimator,
        logger=logger
    )

    # evaluate
    print('This is the performance of the best test model:')
    checkpoint = os.path.join(save_path, 'best_test_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)


if __name__ == '__main__':
    main()
