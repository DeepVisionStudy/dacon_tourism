import os
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoFeatureExtractor

from dataset import create_data_loader_test
from model import TourClassifier, TourClassifier_Continuous, TourClassifier_Separate
from utils import set_seeds, load_config


PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')
PATH_SUBMIT = osp.join(PATH_BASE, 'ensemble')
os.makedirs(PATH_SUBMIT, exist_ok=True)

df = pd.read_csv(osp.join(PATH_DATA, 'train.csv'))

cat1_labels = sorted(list(set(df['cat1'].values.tolist())))
cat2_labels = sorted(list(set(df['cat2'].values.tolist())))
cat3_labels = sorted(list(set(df['cat3'].values.tolist())))

cat1_to_cat2_dict = dict()
for label in cat1_labels:
    tmp = list(set(df[df['cat1'] == label]['cat2'].values))
    cat1_to_cat2_dict[cat1_labels.index(label)] = sorted([cat2_labels.index(i) for i in tmp])

cat1_to_cat3_dict = dict()
for label in cat1_labels:
    tmp = list(set(df[df['cat1'] == label]['cat3'].values))
    cat1_to_cat3_dict[cat1_labels.index(label)] = sorted([cat3_labels.index(i) for i in tmp])

cat2_to_cat1_dict = dict()
for label in cat2_labels:
    tmp = list(set(df[df['cat2'] == label]['cat1'].values))
    cat2_to_cat1_dict[cat2_labels.index(label)] = sorted([cat1_labels.index(i) for i in tmp])

cat2_to_cat3_dict = dict()
for label in cat2_labels:
    tmp = list(set(df[df['cat2'] == label]['cat3'].values))
    cat2_to_cat3_dict[cat2_labels.index(label)] = sorted([cat3_labels.index(i) for i in tmp])

cat3_to_cat1_dict = dict()
for label in cat3_labels:
    tmp = list(set(df[df['cat3'] == label]['cat1'].values))
    cat3_to_cat1_dict[cat3_labels.index(label)] = sorted([cat1_labels.index(i) for i in tmp])

cat3_to_cat2_dict = dict()
for label in cat3_labels:
    tmp = list(set(df[df['cat3'] == label]['cat2'].values))
    cat3_to_cat2_dict[cat3_labels.index(label)] = sorted([cat2_labels.index(i) for i in tmp])


def consider_multi_label(outputs, outputs2, outputs3, cat, dict1, dict2):
    softmax = nn.Softmax(dim=1)
    outputs, outputs2, outputs3 = softmax(outputs), softmax(outputs2), softmax(outputs3)
    
    if cat == 'cat1':
        main, sub1, sub2 = outputs, outputs2, outputs3
    elif cat == 'cat2':
        sub1, main, sub2 = outputs, outputs2, outputs3
    elif cat == 'cat3':
        sub1, sub2, main = outputs, outputs2, outputs3
    
    tmp1 = torch.zeros_like(main)
    for i in range(len(dict1)):
        tmp1[:, i] = sub1[:, dict1[i]].sum(dim=1)
    tmp2 = torch.zeros_like(main)
    for i in range(len(dict2)):
        tmp2[:, i] = sub2[:, dict2[i]].sum(dim=1)
    
    return torch.stack((main, tmp1, tmp2), dim=-1).sum(dim=-1)


def mask_with_before_pred(before_pred, logit, dict):
    output = torch.full(logit.size(), torch.min(logit))
    for i in range(len(before_pred)):
        label = int(before_pred[i])
        for j in dict[label]:
            output[i, j] = logit[i, j]
    return output


def inference(model, data_loader, device, multi_label, mask_label, mode):
    model = model.eval()
    preds_arr, preds_arr2, preds_arr3 = [], [], []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            outputs, outputs2, outputs3 = model(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
            )
            
            if multi_label:
                outputs = consider_multi_label(outputs, outputs2, outputs3, 'cat1', cat1_to_cat2_dict, cat1_to_cat3_dict)
                outputs2 = consider_multi_label(outputs, outputs2, outputs3, 'cat2', cat2_to_cat1_dict, cat2_to_cat3_dict)
                outputs3 = consider_multi_label(outputs, outputs2, outputs3, 'cat3', cat3_to_cat1_dict, cat3_to_cat2_dict)
            
            if mode == 'soft':
                preds_arr.append(np.squeeze(outputs.cpu().numpy()))
                preds_arr2.append(np.squeeze(outputs2.cpu().numpy()))
                preds_arr3.append(np.squeeze(outputs3.cpu().numpy()))
            else:
                _, preds = torch.max(outputs, dim=1)
                if mask_label: outputs2 = mask_with_before_pred(preds, outputs2, cat1_to_cat2_dict)
                _, preds2 = torch.max(outputs2, dim=1)
                if mask_label: outputs3 = mask_with_before_pred(preds2, outputs3, cat2_to_cat3_dict)
                _, preds3 = torch.max(outputs3, dim=1)

                preds_arr.append(np.squeeze(preds.cpu().numpy()))
                preds_arr2.append(np.squeeze(preds2.cpu().numpy()))
                preds_arr3.append(np.squeeze(preds3.cpu().numpy()))

    return preds_arr, preds_arr2, preds_arr3


def save(pred_arr, save_path):
    sample_submission = pd.read_csv(osp.join(PATH_DATA, 'sample_submission.csv'))
    for i in range(len(pred_arr)):
        sample_submission.loc[i, 'cat3'] = cat3_labels[pred_arr[i]]
    sample_submission.to_csv(save_path, index=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test') # valid test soft hard
    parser.add_argument('--path', nargs='+') # work_dirs/exp0/best.pt
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--mask_label', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(args, train_config):
    device = torch.device("cuda:0")
    
    if args.mode == 'valid':
        df = pd.read_csv(osp.join(PATH_DATA, 'train_5fold.csv'))
        df = df[df["kfold"] == train_config.fold].reset_index(drop=True)
    else:
        df = pd.read_csv(osp.join(PATH_DATA, 'test.csv'))

    tokenizer = AutoTokenizer.from_pretrained(train_config.text_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(train_config.image_model)

    eval_data_loader = create_data_loader_test(df, tokenizer, feature_extractor, train_config.max_len)
    if train_config.continuous:
        model = TourClassifier_Continuous
    elif train_config.separate:
        model = TourClassifier_Separate
    else:
        model = TourClassifier
    model = model(
        n_classes1=6, n_classes2=18, n_classes3=128,
        text_model_name=train_config.text_model, image_model_name=train_config.image_model, device=device,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(osp.join(args.work_dir_exp, args.ckpt)))

    preds_arr, preds_arr2, preds_arr3 = inference(model, eval_data_loader, device, args.multi_label, args.mask_label, args.mode)
    
    if args.mode == 'valid':
        for col, arr in zip(['cat1','cat2','cat3'], [preds_arr, preds_arr2, preds_arr3]):
            pred = np.array(arr)
            gt = df[col].values
            acc = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred, average='weighted')
            print(f"[{col}] acc: {round(acc,4)}, f1_acc: {round(f1,4)}")
            # print(df[pred != gt]["id"].to_numpy())
            # print(classification_report(gt, pred))

    elif args.mode == 'test':
        save_path = f'submit_{args.exp}_' + args.ckpt.split('.')[0]
        if args.multi_label: save_path += '_multi'
        if args.mask_label: save_path += '_mask'
        save(preds_arr3, osp.join(args.work_dir_exp, save_path+'.csv'))
    
    elif args.mode in ['soft', 'hard']:
        return preds_arr3


if __name__ == '__main__':
    args = get_parser()
    ensemble = []
    ensemble_file_name = args.mode
    for i in range(len(args.path)):
        path = args.path[i] # work_dirs/exp0/best.pt
        args.exp = path.split('/')[-2] # exp0
        args.ckpt = path.split('/')[-1] # best.pt
        args.work_dir_exp = path[:-len(args.ckpt)-1] # work_dirs/exp0
        
        set_seeds(args.seed)
        train_config = load_config(osp.join(args.work_dir_exp, 'config.yaml'))

        if args.mode in ['soft', 'hard']:
            out = main(args, train_config)
            ensemble.append(out)
            ensemble_file_name += '_' + args.exp + args.ckpt.split('.')[0]
        else:
            main(args, train_config)

    ensemble_save_path = osp.join(PATH_SUBMIT, f'{ensemble_file_name}.csv')
    if args.mode == 'soft':
        preds_arr3 = np.argmax(np.sum(ensemble, axis=0), axis=1)
        save(preds_arr3, ensemble_save_path)
    elif args.mode == 'hard':
        ensemble = np.array(ensemble)
        preds_arr3 = [np.argmax(np.bincount(ensemble[:,i])) for i in range(ensemble.shape[1])]
        save(preds_arr3, ensemble_save_path)