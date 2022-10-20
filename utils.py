import os
import os.path as osp
import time
import math
import yaml
import torch
import random
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import f1_score


def set_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_exp_dir(work_dir):
    work_dir = work_dir.split('./')[-1]
    if not osp.exists(osp.join(os.getcwd(), work_dir)):
        exp_dir = osp.join(os.getcwd(), work_dir, 'exp0')
    else:
        idx = 1
        exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')
        while osp.exists(exp_dir):
            idx += 1
            exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')
    
    os.makedirs(exp_dir)
    return exp_dir


def save_config(args, save_dir):
    with open(save_dir, 'w') as f:
        yaml.safe_dump(args.__dict__, f)


def load_config(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)


def calc_tour_acc(pred, label):
    _, idx = pred.max(1)
    
    acc = torch.eq(idx, label).sum().item() / idx.size()[0] 
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    f1_acc = f1_score(x, y, average='weighted')
    return acc,f1_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))