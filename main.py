# coding: utf-8
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ShopeeModel
from data import ShopeeDataset
import loss
from custom_optimizer import Ranger
from custom_scheduler import ShopeeScheduler
from custom_activation import Mish, replace_activations


parser = argparse.ArgumentParser()
# Data
parser.add_argument('--train_dir', dest='train_dir', default="./train_images/")
parser.add_argument('--train_csv', dest='train_csv', default="./train.csv")
parser.add_argument('--test_dir', dest='test_dir', default="./test_images/")
parser.add_argument('--test_csv', dest='test_csv', default="./test.csv")
parser.add_argument('--submission_dir', dest='submission_dir', default="./public/submission.csv")

# Base
parser.add_argument('--model_dir', dest='model_dir', default="./ckpt/")
parser.add_argument('--resume', dest='resume', default=None)
parser.add_argument('--max_size', dest='max_size', type=int, default=512)
parser.add_argument('--image_size', dest='image_size', type=int, default=512)
parser.add_argument('--epochs', dest='epochs', type=int, default=15)
#parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--seed', dest='batch_size', type=int, default=42)
#parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_workers', dest='num_workers', type=int, default=16)
parser.add_argument('--log_freq', dest='log_freq', type=int, default=10)

# Model Param
parser.add_argument('--model_name', dest='model_name', type=str, default='efficientnet_b5')
parser.add_argument('--n_classes', dest='n_classes', type=int, default=11014)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=512)
parser.add_argument('--use_fc', dest='use_fc', action='store_true')
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.add_argument('--s', dest='s', type=float, default=30)
parser.add_argument('--m', dest='m', type=float, default=0.5)
parser.add_argument('--ls_eps', dest='ls_eps', type=float, default=0.0)
#parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
parser.add_argument('--crit', dest='crit', type=str, default='bce')

# Scheduler Param
parser.add_argument('--lr_start', dest='lr_start', type=float, default=1e-5)
parser.add_argument('--lr_max', dest='lr_max', type=float, default=1e-5 * 16)
parser.add_argument('--lr_min', dest='lr_min', type=float, default=1e-6)
parser.add_argument('--lr_ramp_ep', dest='lr_ramp_ep', type=float, default=5)
parser.add_argument('--lr_sus_ep', dest='lr_sus_ep', type=float, default=0)
parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.8)

args = parser.parse_args()

scheduler_params = {
        "lr_start": args.lr_start,
        "lr_max": args.lr_max * args.batch_size,
        "lr_min": args.lr_min,
        "lr_ramp_ep": args.lr_ramp_ep,
        "lr_sus_ep": args.lr_sus_ep,
        "lr_decay": args.lr_decay,
    }


def train(model, data_loader, optimizer, scheduler, i):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for t, data in enumerate(tk):
        for k, v in data.items():
            data[k] = v.cuda()
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step() 
        fin_loss += loss.item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)),
                        'LR' : optimizer.param_groups[0]['lr']})

    scheduler.step()
    return fin_loss / len(data_loader)


def eval(model, data_loader, i):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(i+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(Config.DEVICE)
            _, loss = model(**data)
            fin_loss += loss.item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
        return fin_loss / len(data_loader)


def run_train():
    df = pd.read_csv(args.train_csv)

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])
    
    # Augmentation
    train_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=120, p=0.8),
        #A.Cutout(p=0.5),
        #A.OneOf([
        #    A.HueSaturationValue(),
        #    A.ShiftScaleRotate()
        #], p=1),
        A.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0),
    ])

    test_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        #A.CenterCrop(args.image_size, args.image_size, p=1.),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])

    # Dataset, Dataloader
    train_dataset = ShopeeDataset(df, data_dir=args.train_dir, transforms=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle = True,
                              drop_last=True)

    model = ShopeeModel(model_name=args.model_name,
                        n_classes=args.n_classes,
                        fc_dim=args.feat_dim,
                        scale=args.s,
                        margin=args.m,
                        #crit=args.crit,
                        use_fc=args.use_fc,
                        pretrained=args.pretrained)
    model.cuda()
    
    existing_layer = torch.nn.SiLU
    new_layer = Mish()
    # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()
    model = replace_activations(model, existing_layer, new_layer)
    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.resume)))
    
    optimizer = Ranger(model.parameters(), lr = scheduler_params['lr_start'])
    scheduler = ShopeeScheduler(optimizer, **scheduler_params)

    for i in range(args.epochs):
        avg_loss_train = train(model, train_loader, optimizer, scheduler, i)
        torch.save(model.state_dict(), 
                   os.path.join(args.model_dir, f'arcface_512x512_{args.model_name}_epoch{i+1}.pt'))


if __name__ == '__main__':
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    # make directory
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    
    run_train()

