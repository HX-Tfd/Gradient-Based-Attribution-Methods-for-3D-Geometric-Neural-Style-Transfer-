import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ShapeNetDataset
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
global variables
'''
num_points = 2048
bs = 128
model_path = "pretrained_models"

def test_class_accuracy(ckpt_path, by_class=False, class_name=None):
    if by_class:
        assert class_name is not None

    if by_class:
        name2id = {class_name: 0} # can be a list
    else:
        name2id = {name: id for id, name in enumerate(list(get_shapename2id().keys()), 0)}
    class_corr, class_total = np.zeros(len(name2id)), np.zeros(len(name2id))

    # load model and data
    model_dict = os.path.join(model_path, ckpt_path)
    model = load_model(path=model_dict, device=device, option='classification', k=len(get_shapename2id()),
                       global_pooling='max',
                       use_clf=True,
                       new_ckpt=False,
                       retain_feature_grad=False
                       )

    if by_class:
        print("testing accuracy on class", class_name)

    test_dataset = ShapeNetDataset(root="data",
                                   split='train',
                                   split_dir='train_test_split',
                                   shapename2id=get_shapename2id(),
                                   num_points=num_points,
                                   with_augmentation=True,
                                   shape_class=class_name if by_class else None)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=True
    )

    # run test
    print("running test set")
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(test_dataloader, 0), desc="testing ..."):
        points, label = data
        points = points.to(device)
        label = label.to(device)
        model = model.eval()
        pred_scores = model(points)['scores']
        pred_classes = pred_scores.data.max(1)[1] # [B, num_classes]

        for j in range(len(pred_classes)):
            idx = 0 if by_class else pred_classes[j]
            if pred_classes[j] == label[j]:
                total_correct += 1
                class_corr[idx] += 1
            class_total[idx] += 1
        total_testset += points.size()[0]
        if i > 20:
            break

    acc_per_class = class_corr/class_total
    class_accuracy = {name: "{} ({}/{})".format(acc, int(corr), int(total)) for name, acc, corr, total in
                      zip(list(name2id.keys()), acc_per_class, class_corr, class_total)}

    print("accuracy per class: ", class_accuracy)
    #plot
    acc_per_class[np.isnan(acc_per_class)] = 0
    plt.xticks(range(0, len(name2id)), list(name2id.keys()), rotation=30)
    xlocs, xlabs = plt.xticks()
    plt.bar(range(0, len(name2id)), acc_per_class)
    for i, v in enumerate(acc_per_class):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 3))) #TODO: needs adaption

    plt.xlabel("class names")
    plt.ylabel("accuracy")
    plt.show()

    if by_class:
        return acc_per_class
    else:
        return class_accuracy



def select_best_model():
    acc = []
    models = [os.path.join(model_path, path) for path in os.listdir(model_path) if path.endswith('.pth')]
    for m in range(len(models)):
        model_dict = models[m]
        model=load_model(path=model_dict, device=device, option='classification', k=len(get_shapename2id()),
                         global_pooling='max',
                         use_clf=True,
                         new_ckpt=False,
                         retain_feature_grad=False
                         )

        test_dataset = ShapeNetDataset(root="data",
                                       split='test',
                                       shapename2id=get_shapename2id(),
                                       num_points=num_points,
                                       with_augmentation=True)

        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=bs,
            shuffle=True
        )

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

        acc.append(total_correct / float(total_testset))

        print("model {} accuracy: {}".format(m, acc[m]))

    print("model {} has the highest accuracy. See {}".format(np.argmax(acc), models[np.argmax(acc)]))


if __name__ == "__main__":
    test_class_accuracy(ckpt_path="clf_ep_94_aug.pth")