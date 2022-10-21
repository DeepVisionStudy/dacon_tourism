import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from transformers import AutoTokenizer, AutoFeatureExtractor

from dataset import create_data_loader_test
from model import TourClassifier
from utils import set_seeds, load_config


PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')


def inference(model, data_loader, device):
    model = model.eval()
    preds_arr = []
    preds_arr2 = []
    preds_arr3 = []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            outputs, outputs2, outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds = torch.max(outputs, dim=1)
            _, preds2 = torch.max(outputs2, dim=1)
            _, preds3 = torch.max(outputs3, dim=1)

            preds_arr.append(preds.cpu().numpy()[0])
            preds_arr2.append(preds2.cpu().numpy()[0])
            preds_arr3.append(preds3.cpu().numpy()[0])

    return preds_arr, preds_arr2, preds_arr3


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    args.work_dir_exp = f'./work_dirs/exp{args.exp}'
    args.ckpt_dir = osp.join(args.work_dir_exp, f'epoch{args.epoch}.pt')
    return args


def main(args, train_config):
    train_df = pd.read_csv(osp.join(PATH_DATA, 'train.csv'))
    cat3_labels = sorted(list(set(train_df['cat3'].values.tolist())))

    device = torch.device("cuda:0")
    
    if args.mode == 'valid':
        df = pd.read_csv(osp.join(PATH_DATA, 'train_5fold.csv'))
        df = df[df["kfold"] == train_config.fold].reset_index(drop=True)
    elif args.mode == 'test':
        df = pd.read_csv(osp.join(PATH_DATA, 'test.csv'))

    tokenizer = AutoTokenizer.from_pretrained(train_config.text_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(train_config.image_model)

    eval_data_loader = create_data_loader_test(df, tokenizer, feature_extractor, train_config.max_len)

    model = TourClassifier(
        n_classes1=6, n_classes2=18, n_classes3=128,
        text_model_name=train_config.text_model, image_model_name=train_config.image_model, device=device,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt_dir))

    preds_arr, preds_arr2, preds_arr3 = inference(model, eval_data_loader, device)
    
    if args.mode == 'valid':
        for col, arr in zip(['cat1','cat2','cat3'], [preds_arr, preds_arr2, preds_arr3]):
            pred = np.array(arr)
            gt = df[col].values
            acc = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred, average='weighted')
            print(f"[{col}] acc: {round(acc,4)}, f1_acc: {round(f1,4)}")
            # print(classification_report(gt, pred))

    elif args.mode == 'test':
        sample_submission = pd.read_csv(osp.join(PATH_DATA, 'sample_submission.csv'))
        for i in range(len(preds_arr3)):
            sample_submission.loc[i, 'cat3'] = cat3_labels[preds_arr3[i]]
        sample_submission.to_csv(
            osp.join(args.work_dir_exp, f'submit_exp{args.exp}_epoch{args.epoch}.csv'), index=False)


if __name__ == '__main__':
    args = get_parser()
    set_seeds(args.seed)
    train_config = load_config(osp.join(args.work_dir_exp, 'config.yaml'))
    main(args, train_config)