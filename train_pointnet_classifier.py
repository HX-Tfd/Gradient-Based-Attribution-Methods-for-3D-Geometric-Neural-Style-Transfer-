"""
Modified from:
 https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
"""

import time
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import open3d as o3d

from utils import *
from modules.loss import feature_transform_regularizer

from modules.pointnet_classifier import PointNetClf
import shapenet_constants
from dataset import ShapeNetDataset

"""
global configs
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# reproduce original PointNet paper
orth_regularization = True # only for the feature transformer
regularization_weight = 0.001
bn_decay_start = 0.5
bn_rate_max = 0.99
optim_lr_init = 0.001
optim_momentum = 0.9
batch_size = 64 # 32
optim_lr_dec_rate, optim_lr_dec_per = 0.5, 10 #.5, 20

# training configs
num_epochs = 100 # 250
save_every = 20
eval_every = 10

# data configs
num_points = 2048
data_path = "data"
shape_classes = ["airplane", "car", "chair", "lamp", "table"]

# parallel configs
num_workers = 1

# IO configs
save_path = "logs/train_classifier_test"
writer = SummaryWriter(log_dir=os.path.join(save_path, 'summary'))


"""
Data preparation
"""

# load model
model = PointNetClf(
    k=len(get_shapename2id()),
    with_transformer=True,
    global_pooling='max',
    use_clf=True
).to(device)

# load data
print("loading data")
train_dataset = ShapeNetDataset(root=data_path,
                               split='train',
                               split_dir='train_test_split_imbal',
                               shapename2id=get_shapename2id(),
                               num_points=num_points,
                               with_augmentation=True,
                                shape_class=shape_classes)

test_dataset = ShapeNetDataset(root=data_path,
                               split='test',
                               split_dir='train_test_split_imbal',
                               shapename2id=get_shapename2id(),
                               num_points=num_points,
                               with_augmentation=True,
                               shape_class=shape_classes)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
    #num_workers=int(num_workers)
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
    #num_workers=int(num_workers)
)

print("set len:", len(train_dataset))
num_batches = len(train_dataset)/batch_size

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr_init, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_lr_dec_per, gamma=optim_lr_dec_rate)

"""
Training
"""
#TODO: try dense clf or larger batchsize, lrdec, momentum in BatchNorm layers
print("running training set")
for epoch in tqdm(range(1, num_epochs+1), desc="training epoch ..."):
    for i, data in enumerate(train_dataloader, 0):
        points, label = data
        points = points.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        model = model.train()

        out_dict = model(points)
        pred_scores, transformed_feats = out_dict['scores'], out_dict['transformed_features']
        loss = F.nll_loss(pred_scores, label)
        if orth_regularization:
            regularization_loss = regularization_weight * feature_transform_regularizer(transformed_feats[1])
            loss += regularization_loss
        loss.backward()
        optimizer.step()

        pred_classes = pred_scores.data.max(1)[1]
        correct = pred_classes.eq(label.data).cpu().sum()
        print('[Epoch %d: batch %d/%d] train loss: %f accuracy: %f' %
              (epoch, i, num_batches, loss.item(), correct.item() / float(batch_size)))

        # summary
        writer.add_scalar("{}/total clf loss".format("total_loss"),
                          loss, i + epoch * len(train_dataset) / float(batch_size))
        writer.add_scalar("{}/regularization loss".format("regl_loss"),
                          regularization_loss, i + epoch * len(train_dataset) / float(batch_size))

        # evaluation
        if (i+1) % eval_every == 0:
            j, data = next(enumerate(test_dataloader, 0))
            points, label = data
            points = points.to(device)
            label = label.to(device)
            model = model.eval()
            pred_scores = model(points)['scores']
            loss = F.nll_loss(pred_scores, label)
            pred_classes = pred_scores.data.max(1)[1]
            correct = pred_classes.eq(label.data).cpu().sum()
            print('[Epoch %d TEST] test loss: %f test accuracy: %f' %
                  (epoch, loss.item(), correct.item() / float(batch_size)))
            writer.add_scalar("{}/accuracy".format("eval"),
                              correct.item() / float(batch_size),
                              i + epoch * len(train_dataset) / float(batch_size))
    scheduler.step()

    if epoch % save_every == 0:
        torch.save(model.state_dict(), '{}/clf_ep_{}.pth'.format(save_path, epoch))


print("running test set")
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(test_dataloader, 0), desc="testing ..."):
    points, label = data
    points = points.to(device)
    label = label.to(device)
    model = model.eval()
    pred_scores = model(points)['scores']
    pred_classes = pred_scores.data.max(1)[1]
    correct = pred_classes.eq(label.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy: {}".format(total_correct / float(total_testset)))

