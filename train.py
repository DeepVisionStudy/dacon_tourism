import time
import wandb
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers.optimization import get_cosine_schedule_with_warmup
from adamp import AdamP

from dataset import create_data_loader
from loss import FocalLoss
from model import TourClassifier, TourClassifier_Separate
from utils import set_seeds, get_exp_dir, save_config, AverageMeter, calc_tour_acc, timeSince
from set_wandb import wandb_init


PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')


def get_loss_coef(loss1, loss2, loss3):
    l1, l2, l3 = loss1.item(), loss2.item(), loss3.item()
    loss_coef = 1.0
    if l1==0 and l2!=0 and l3==0: # O X O
        loss_coef = 1.5
    elif l1!=0:
        if l2==0:
            if l3==0:             # X O O
                loss_coef = 1.5
            else:                 # X O X
                loss_coef = 2.0
        elif l3==0:               # X X O
            loss_coef = 2.0
    return loss_coef


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples, epoch,
                device, lambda_cat1, lambda_cat2, lambda_cat3, val_freq, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()
    sent_count = AverageMeter()
    start = end = time.time()

    model = model.train()
    # wandb.watch(model)
    correct_predictions = 0
    for step, d in enumerate(data_loader):
        data_time.update(time.time() - end)
        batch_size = d["input_ids"].size(0)

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device)
        # pixel_values = torch.zeros_like(d['pixel_values']).to(device)
        
        cats1 = d["cats1"].to(device)
        cats2 = d["cats2"].to(device)
        cats3 = d["cats3"].to(device)
        
        with autocast():
            outputs, outputs2, outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)

            loss1 = loss_fn(outputs, cats1)
            loss2 = loss_fn(outputs2, cats2)
            loss3 = loss_fn(outputs3, cats3)
            loss = loss1 * lambda_cat1 + loss2 * lambda_cat2 + loss3 * lambda_cat3
            # loss_coef = get_loss_coef(loss1, loss2, loss3)
            # loss = loss1 * (lambda_cat1*loss_coef) + loss2 * (lambda_cat2*loss_coef) + loss3 * (lambda_cat3*loss_coef)
            losses.update(loss.item(), batch_size)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        # scheduler.step()
        # optimizer.zero_grad()

        correct_predictions += torch.sum(preds == cats3)
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        if step % val_freq == 0 or step+1 == len(data_loader):
            acc,f1_acc = calc_tour_acc(outputs3, cats3)
            accuracies.update(acc, batch_size)
            f1_accuracies.update(f1_acc, batch_size)

            print(
                'Epoch: [{0}][{1}/{2}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                'Acc: {acc.val:.3f}({acc.avg:.3f}) '
                'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) '
                'sent/s {sent_s:.0f} '
                .format(
                    epoch, step+1, len(data_loader),
                    data_time=data_time,
                    remain=timeSince(start, float(step+1)/len(data_loader)),
                    loss=losses,
                    acc=accuracies,
                    f1_acc=f1_accuracies,
                    sent_s=sent_count.avg / batch_time.avg
                )
            )

    wandb.log({
        'train/loss': round(losses.avg,4), 'train/acc': round(accuracies.avg,4), 'train/f1': round(f1_accuracies.avg,4)
    }, commit=False)

    return correct_predictions.double() / n_examples, losses.avg


def validate(model, data_loader, loss_fn, device, lambda_cat1, lambda_cat2, lambda_cat3):
    model = model.eval()
    losses = []
    cnt = 0
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            cats1 = d["cats1"].to(device)
            cats2 = d["cats2"].to(device)
            cats3 = d["cats3"].to(device)
            outputs, outputs2, outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            loss1 = loss_fn(outputs, cats1)
            loss2 = loss_fn(outputs2, cats2)
            loss3 = loss_fn(outputs3, cats3)
            loss = loss1 * lambda_cat1 + loss2 * lambda_cat2 + loss3 * lambda_cat3
            # loss_coef = get_loss_coef(loss1, loss2, loss3)
            # loss = loss1 * (lambda_cat1*loss_coef) + loss2 * (lambda_cat2*loss_coef) + loss3 * (lambda_cat3*loss_coef)
            losses.append(loss.item())

            if cnt == 0:
                cnt += 1
                outputs_arr, outputs2_arr, outputs3_arr = outputs, outputs2, outputs3
                cats1_arr, cats2_arr, cats3_arr = cats1, cats2, cats3
            else:
                outputs_arr = torch.cat([outputs_arr, outputs], 0)
                outputs2_arr = torch.cat([outputs2_arr, outputs2], 0)
                outputs3_arr = torch.cat([outputs3_arr, outputs3], 0)
                cats1_arr = torch.cat([cats1_arr, cats1], 0)
                cats2_arr = torch.cat([cats2_arr, cats2], 0)
                cats3_arr = torch.cat([cats3_arr, cats3], 0)

    acc, f1_acc = calc_tour_acc(outputs_arr, cats1_arr, report=True)
    acc2, f1_acc2 = calc_tour_acc(outputs2_arr, cats2_arr, report=True)
    acc3, f1_acc3 = calc_tour_acc(outputs3_arr, cats3_arr, report=True)

    wandb.log({
        'valid/loss': round(np.mean(losses),4), 'valid/acc': round(acc3,4), 'valid/f1': round(f1_acc3,4),
        'valid/acc_cat1': round(acc,4), 'valid/f1_cat1': round(f1_acc,4),
        'valid/acc_cat2': round(acc2,4), 'valid/f1_cat2': round(f1_acc2,4),
    })

    return acc3, f1_acc3, np.mean(losses)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--val_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--work_dir', type=str, default='./work_dirs')
    
    parser.add_argument('--df_ver', type=int, default=1)

    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--separate_alpha', type=float, default=0.2)
    
    parser.add_argument('--text_model', type=str, default="klue/roberta-base")
    parser.add_argument('--image_model', type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--lambda_cat1', type=float, default=0.05)
    parser.add_argument('--lambda_cat2', type=float, default=0.1)
    parser.add_argument('--lambda_cat3', type=float, default=0.85)
    parser.add_argument('--lambda_image_lr', type=float, default=1)
    parser.add_argument('--lambda_linear_lr', type=float, default=1)
    
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lambda_text_wd', type=float, default=1)
    parser.add_argument('--lambda_image_wd', type=float, default=1)

    parser.add_argument('--warmup_cycle', type=int, default=2)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    args = parser.parse_args()

    args.work_dir_exp = get_exp_dir(args.work_dir)
    args.text_model_name = args.text_model.split('/')[-1]
    args.image_model_name = args.image_model.split('/')[-1]
    args.config_dir = osp.join(args.work_dir_exp, 'config.yaml')
    return args


def main(args):
    args.device = torch.device("cuda:0")

    if args.df_ver == 1:
        df = 'train_5fold.csv'
    elif args.df_ver == 2:
        df = 'train_5fold_ver2.csv'
    df = pd.read_csv(osp.join(PATH_DATA, df))
    train = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.image_model)

    train_data_loader = create_data_loader(
        train, tokenizer, feature_extractor, args.max_len, args.batch_size, args.num_workers, mode='train')
    valid_data_loader = create_data_loader(
        valid, tokenizer, feature_extractor, args.max_len, args.batch_size, args.num_workers, mode='valid')

    if args.separate:
        model = TourClassifier_Separate(
            n_classes1=6, n_classes2=18, n_classes3=128,
            text_model_name=args.text_model, image_model_name=args.image_model, device=args.device,
            dropout=args.dropout, alpha=args.separate_alpha,
        ).to(args.device)
    else:
        model = TourClassifier(
            n_classes1=6, n_classes2=18, n_classes3=128,
            text_model_name=args.text_model, image_model_name=args.image_model, device=args.device,
            dropout=args.dropout,
        ).to(args.device)

    # loss_fn = nn.CrossEntropyLoss().to(args.device)
    loss_fn = FocalLoss().to(args.device)

    lr = args.learning_rate
    lr_linear = lr * args.lambda_linear_lr
    lr_image = lr * args.lambda_image_lr
    lr_linear_image = lr_linear * args.lambda_image_lr
    params = [
        {"params": model.text_model.parameters(), "lr": lr, "weight_decay": args.weight_decay*args.lambda_text_wd},
        {"params": model.text_cls.parameters(), "lr": lr_linear, "weight_decay": args.weight_decay},
        {"params": model.text_cls2.parameters(), "lr": lr_linear, "weight_decay": args.weight_decay},
        {"params": model.text_cls3.parameters(), "lr": lr_linear, "weight_decay": args.weight_decay},
        {"params": model.image_model.parameters(), "lr": lr_image, "weight_decay": args.weight_decay*args.lambda_image_wd},
        {"params": model.image_cls.parameters(), "lr": lr_linear_image, "weight_decay": args.weight_decay},
        {"params": model.image_cls2.parameters(), "lr": lr_linear_image, "weight_decay": args.weight_decay},
        {"params": model.image_cls3.parameters(), "lr": lr_linear_image, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(params)
    # optimizer = AdamP(model.parameters(), lr=args.learning_rate)
    
    scaler = GradScaler()

    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
        num_cycles=(1/args.warmup_cycle),
    )

    max_f1 = 0
    for epoch in range(1, args.epochs + 1):
        print('-' * 10)
        print(f'Epoch {epoch}/{args.epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, scheduler, len(train), epoch, args.device,
            args.lambda_cat1, args.lambda_cat2, args.lambda_cat3, args.val_freq, scaler,
        )
        validate_acc, validate_f1, validate_loss = validate(
            model, valid_data_loader, loss_fn, args.device,
            args.lambda_cat1, args.lambda_cat2, args.lambda_cat3,
        )
        if validate_f1 > max_f1:
            max_f1 = validate_f1
            torch.save(model.state_dict(), osp.join(args.work_dir_exp, 'best.pt'))
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), osp.join(args.work_dir_exp, f'epoch{epoch}.pt'))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Validate loss {validate_loss} accuracy {validate_acc} f1 {validate_f1}')
        print("")


if __name__ == '__main__':
    for fold in [0,1,2,3,4]:
        args = get_parser()
        args.fold = fold
        save_config(args, args.config_dir)
        set_seeds(args.seed)
        wandb_init(args)
        main(args)