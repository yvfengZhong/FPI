import os
import random
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np

from utils.func import *
from scripts.train import evaluate
from utils.metrics import Estimator
from data.builder import get_imgs_labels_path, get_mean_and_std, get_dataset
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
    save_path = os.path.join(cfg.base.log_path, cfg.train.network) if not cfg.model.dual else os.path.join(cfg.base.log_path, "attention")
    cfg.train.checkpoint = save_path
    sys.stdout = Logger(os.path.join(save_path, 'log_test.txt'))

    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
    data_root = os.path.join(root_path, "dataset")
    scaler_path = os.path.join(root_path, "checkpoints/mlp")

    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # test
    print("testing beginning!")
    set_random_seed(cfg.base.random_seed)

    # model
    model = generate_model(cfg) if not cfg.model.dual else generate_dual_model(cfg)

    # data
    imgs_path, seqs, labels_path = get_imgs_labels_path(data_root, split='train_val', data_name=cfg.data.name, single=cfg.data.single, scaler_path=scaler_path)
    get_mean_and_std(cfg, imgs_path, seqs, labels_path)

    imgs_path, seqs, labels_path = get_imgs_labels_path(data_root, split='test', data_name=cfg.data.name, single=cfg.data.single, scaler_path=scaler_path)
    test_dataset = get_dataset(cfg, imgs_path, seqs, labels_path, split='test')

    # estimator
    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)

    # evaluate
    print('This is the performance of the best test model:')
    checkpoint = os.path.join(cfg.train.checkpoint, 'best_test_weights.pt') if not cfg.model.dual else os.path.join(cfg.train.finetune, 'best_test_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)


if __name__ == '__main__':
    main()
