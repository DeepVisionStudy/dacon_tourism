import argparse
import pandas as pd
import os.path as osp
from tqdm import tqdm

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

            preds_arr.append(preds.cpu().numpy())
            preds_arr2.append(preds2.cpu().numpy())
            preds_arr3.append(preds3.cpu().numpy())

    return preds_arr, preds_arr2, preds_arr3


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()
    args.work_dir_exp = f'./work_dirs/exp{args.exp}'
    return args


def main(args):
    train_df = pd.read_csv(osp.join(PATH_DATA, 'train.csv'))
    cat3_labels = sorted(list(set(train_df['cat3'].values.tolist())))

    device = torch.device("cuda:0")
    df = pd.read_csv(osp.join(PATH_DATA, 'test.csv'))

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.image_model)

    eval_data_loader = create_data_loader_test(df, tokenizer, feature_extractor, args.max_len)

    model = TourClassifier(
        n_classes1=6, n_classes2=18, n_classes3=128,
        text_model_name=args.text_model, image_model_name=args.image_model, device=device
    ).to(device)

    preds_arr, preds_arr2, preds_arr3 = inference(model, eval_data_loader, device)

    sample_submission = pd.read_csv(osp.join(PATH_DATA, 'sample_submission.csv'))
    for i in range(len(preds_arr3)):
        sample_submission.loc[i,'cat3'] = cat3_labels[preds_arr3[i][0]]

    sample_submission.to_csv(osp.join(args.work_dir_exp, 'submission.csv'), index=False)


if __name__ == '__main__':
    args = get_parser()
    args = load_config(osp.join(args.work_dir_exp, 'config.yaml'))
    set_seeds(args.seed)
    main(args)