import os

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import random
import open3d as o3d
import matplotlib.pyplot as plt

from utils import *


def analyze_classes_txt():
    data_dir = "data/train_test_split"
    y = {name: 0 for name in list(get_shapename2id().keys())}
    with open('{}/train_data.txt'.format(data_dir), 'r') as f_train, \
            open('{}/test_data.txt'.format(data_dir), 'r') as f_test:
        for line in f_train.readlines():
            _, class_name = line.split()
            y[class_name] += 1
        for line in f_test.readlines():
            _, class_name = line.split()
            y[class_name] += 1
    y = [item[1] for item in sorted(y.items())]

    namelist = list(get_shapename2id().keys())
    plt.xticks(range(0, len(namelist)), namelist, rotation=30)
    xlocs, xlabs = plt.xticks()
    plt.bar(range(0, len(namelist)), y)
    for i, v in enumerate(y):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 3)))
    plt.xlabel("class names")
    plt.ylabel("number of objects")
    plt.show()


def analyze_classes(plot=True):
    """
    Analyze and optionally plots the number of the classes in the dataset
    :returns a dictionary of {class name: num objects in the class}
    """
    data_dir = "data/PartAnnotation"
    ids = [d for d in os.listdir(data_dir) if not (d.startswith('.') or
                                                 (os.path.isfile(os.path.join(data_dir, d))))]
    shapeid2name = {sid: name for name, sid in get_shapename2id().items()}
    y = {}
    for dir in ids:
        pts = os.listdir(os.path.join(data_dir, dir, "points"))
        num_objs = len([name for name in pts if os.path.isfile(os.path.join(data_dir, dir, "points", name))])
        y[shapeid2name[dir]] = num_objs
    y = [ item[1] for item in sorted(y.items())]

    namelist = list(get_shapename2id().keys())
    if plot:
        plt.xticks(range(0, len(namelist)), namelist, rotation=30)
        xlocs, xlabs = plt.xticks()
        plt.bar(range(0, len(namelist)), y)
        for i, v in enumerate(y):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(round(v, 3)))
        plt.xlabel("class names")
        plt.ylabel("number of objects")
        plt.show()

    return {class_name: num_objs for class_name, num_objs in zip(namelist, y)}


def data_rebalance(root="./data", split=0.7):
    """
    Creates a balanced train test split in ./data/PartAnnotation/train_test_split
    Apply random undersampling and upsampling with optional augmentation
    """
    data_root = "PartAnnotation"
    classid2name = {v: k for k, v in get_shapename2id().items()}
    num_obj_per_class = 1500
    classname2numobj = analyze_classes(plot=False)

    split_dir = os.path.join(root, 'train_test_split')
    if not os.path.exists(split_dir):
        os.system("mkdir {}".format(split_dir))

    with open('{}/train_data.txt'.format(split_dir), 'r+') as f_train, \
            open('{}/test_data.txt'.format(split_dir), 'r+') as f_test:

        data_path = os.path.join(root, data_root)
        train_list, test_list = [], []
        for shape_dir in os.listdir(data_path):
            if shape_dir.startswith('.') or (not os.path.isdir(os.path.join(data_path, shape_dir))):
                print("SKIPPED {}".format(shape_dir))
                continue
            print("processing ", shape_dir)
            obj_id_list = np.array(os.listdir(os.path.join(data_path, shape_dir, 'points')))
            shape_class = classid2name[shape_dir]

            # apply undersampling
            if classname2numobj[shape_class] >= num_obj_per_class:
                ids = np.random.choice(len(obj_id_list), num_obj_per_class, replace=False)
                obj_id_list = obj_id_list[ids]

            # apply oversampling by duplicating
            else:
                orig_list = obj_id_list
                while len(obj_id_list) < num_obj_per_class:
                    obj_id_list = np.concatenate((obj_id_list, orig_list))

            split_id = int(len(obj_id_list) * split)
            for i in range(0, split_id):
                train_list.append("{} {}\n".format(obj_id_list[i], shape_class))
            for i in range(split_id + 1, len(obj_id_list)):
                test_list.append("{} {}\n".format(obj_id_list[i], shape_class))

        for line in train_list:
            f_train.write(line)
        for line in test_list:
            f_test.write(line)


def create_train_test_split_seg(root="./data", split=0.7, shuffle=False):
    """
    creates train and test splits within the directory root/train_test_split
    split = 0.7  # train 0.7, test 0.3

    this uses the .pts files from PartAnnotation

    default: root = . (./data)
    """
    data_root = "PartAnnotation"
    classid2name = {v: k for k, v in get_shapename2id().items()}

    split_dir = os.path.join(root, 'train_test_split')
    if not os.path.exists(split_dir):
        os.system("mkdir {}".format(split_dir))

    with open('{}/train_data.txt'.format(split_dir), 'r+') as f_train, \
            open('{}/test_data.txt'.format(split_dir), 'r+') as f_test:

        data_path = os.path.join(root, data_root)
        train_list, test_list = [], []
        for shape_dir in os.listdir(data_path):
            if shape_dir.startswith('.') or (not os.path.isdir(os.path.join(data_path, shape_dir))):
                print("SKIPPED {}".format(shape_dir))
                continue
            print("processing ", shape_dir)
            obj_id_list = os.listdir(os.path.join(data_path, shape_dir, 'points'))
            split_id = int(len(obj_id_list) * split)
            shape_class = classid2name[shape_dir]

            for i in range(0, split_id - 1):
                train_list.append("{} {}\n".format(obj_id_list[i], shape_class))

            for i in range(split_id, len(obj_id_list)):
                test_list.append("{} {}\n".format(obj_id_list[i], shape_class))

        for line in train_list:
            f_train.write(line)
        for line in test_list:
            f_test.write(line)


def create_train_test_split(root="./data", split=0.7, shuffle=True):
    """
    creates train and test splits within the directory root/train_test_split
    split = 0.7  # train 0.7, test 0.3

    default: root = . (./data)
    """
    data_root = "ShapeNetCore.v2"

    split_dir = os.path.join(root, 'train_test_split')
    if not os.path.exists(split_dir):
        os.system("mkdir {}".format(split_dir))

    with open('{}/train_data.txt'.format(split_dir), 'r+') as f_train, \
            open('{}/test_data.txt'.format(split_dir), 'r+') as f_test:
        # create shape_id-shape_class pairs
        data_path = os.path.join(root, data_root)
        train_list, test_list = [], []
        for shape_dir in os.listdir(data_path):
            if shape_dir.startswith('.'):
                print("SKIPPED {}".format(shape_dir))
                continue
            print("processing ", shape_dir)
            shape_class = shape_dir
            obj_id_list = os.listdir(os.path.join(data_path, shape_dir))
            split_id = int(len(obj_id_list) * split)

            for i in range(0, split_id-1):
                train_list.append("{} {}\n".format(obj_id_list[i], shape_class))

            for i in range(split_id, len(obj_id_list)):
                test_list.append("{} {}\n".format(obj_id_list[i], shape_class))

        # shuffle data
        if shuffle:
            random.shuffle(train_list)
            random.shuffle(test_list)

        for line in train_list:
            f_train.write(line)
        for line in test_list:
            f_test.write(line)


class ShapeNetDataset(data.Dataset):
    """
    A class representing (a subset of) the ShapeNet dataset

    A training (input, output) pair is a tuple of (shape_id, shape_class)

    Create a folder "train_test_split" in the root directory and put/generate
    your train/test splits there

    :param root: the root containing the data (default ./data)
    :param split: one of ['train', 'test']
    :param class_list: a list of the class names
    :param shape_class: a string or a list of string representing the class(es)
    """
    data_root = "PartAnnotation"

    def __init__(self, root, split, split_dir, shapename2id, num_points, with_augmentation, shape_class=None):
        self.root = root
        self.split = split
        self.split_dir = split_dir
        self.shapename2id = shapename2id
        #self.class_id = list(shapename2id.values())
        self.num_points = num_points
        self.with_augmentation = with_augmentation
        self.shapeid2class = []

        if isinstance(shape_class, list):
            self.cln = [c.upper() for c in shape_class]
            self.class_dict = {name: id for id, name in
                               enumerate([shapename for shapename in list(shapename2id.keys()) if shapename in self.cln], 0)}
        elif isinstance(shape_class, str):
            self.cln = shape_class
            self.class_dict = {name: id for id, name in enumerate(list(shapename2id.keys()), 0)}
        else:
            raise IOError("shape name must be a string or a list of strings!")


        # if there is no split file, create a split file first
        split_file = os.path.join(self.root, self.split_dir, '{}_data.txt'.format(split))
        with open(split_file, 'r') as f:
            for line in f.readlines():
                shape_id, class_name = line.split()
                if self.cln is not None:
                    if isinstance(self.cln, str):
                        if class_name == self.cln.upper():
                            self.shapeid2class.append((shape_id, class_name))
                    elif isinstance(self.cln, list):
                        if class_name in self.cln:
                            self.shapeid2class.append((shape_id, class_name))
                else:
                    self.shapeid2class.append((shape_id, class_name))

    def __getitem__(self, idx):
        id2class = self.shapeid2class[idx]
        shape_id, class_name = id2class[0], id2class[1]

        # input, subdivide if the object has less than self.num_points vertices
        #data_path = os.path.join(self.root, self.data_root, class_name, shape_id, 'models/model_normalized.obj')
        data_path = os.path.join(self.root, self.data_root, self.shapename2id[class_name], 'points', shape_id)
        points = get_samples(path=data_path, d_type='txt', with_normalization=True, to_tensor=False,
                             subdivide=True, num_vert=self.num_points)
        points = sample_uniform_from(points, num_samples=self.num_points, replace=True)

        #standardize [3, N]
        points -= points.mean(1, keepdims=True)
        points /= np.max(np.sqrt(np.sum(points**2, axis=0)))

        if self.with_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            points[[0, 2], :] = rotation_matrix.transpose(0, 1).dot(points[[0, 2], :])  # random rotation
            points += np.random.normal(0, 0.02, size=points.shape)  # random jitter
        points = torch.from_numpy(points).float()

        # label
        class_label = torch.tensor(self.class_dict[class_name])

        return points, class_label

    def __len__(self):
        return len(self.shapeid2class)


def main():
    data_rebalance()


if __name__ == '__main__':
    main()