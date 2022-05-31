import time
import os, sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import open3d as o3d

from utils import *
import shapenet_constants
from guided_bp import GuidedBackprop
import integrated_gradients

device = "cuda" if torch.cuda.is_available() else "cpu"


# deep dream algorithm
def gradient_step(model, inp, layer_ids, lr, iteration, orig_inp=None, **kwargs):
    """

    :param model: the feature extractor
    :param inp: the target shape of mixture
    :param layer_ids: ids indicating which layers of the feature extractor to use
    :param lr: learning rate
    :param iteration: current iteration
    :param orig_inp: original shape
    :param kwargs:
        -'part_ids': a torch.Tensor of part ids corresponding the used parts of the target shape
        -'use_activation': whether to the feature maps from the model
        -'use_scores': whether to use the part segmentation scores from the model
        -'use_clf_feats'; wheter to use the feature maps from the model's classifier
        -'channel_ids': the ids of the selected feature channels
        -'ascent': if True, does gradient ascent other wise does descent
        -'normalise': wheter to normalise the updated point cloud into [-1, 1]^3
        -'use_ig': whether to use integrated gradients
        caveat: at least one of the 'use_' has to be set to True
    :return: returns the updated shape via deepdream
    """

    # sanity checks
    if kwargs['use_clf_feats']:
        assert kwargs['use_scores']
    else:
        assert kwargs['use_scores'] or kwargs['use_activation']

    features, scores = model(inp)

    # load part scores
    if kwargs['use_scores'] and not kwargs['use_clf_feats']:
        part_classes = torch.max(torch.squeeze(scores), dim=0).indices
        mask = torch.zeros_like(scores)
        for n in range(inp.shape[2]):
            if part_classes[n] == kwargs['part_ids']:
                mask[:, kwargs['part_ids'], n] = 1 # only keep scores of the part corr. to part_ids
        scores = scores * mask # what abt only keep positive scores?
        loss = torch.nn.MSELoss(reduction='mean')(scores, torch.zeros_like(scores)) #resp mean
        loss.backward()
        print(scores[:, kwargs['part_ids'], :])

        grads = (inp.grad)[:, :3, :].detach().clone()  # ?
        grad_std = torch.std(grads)
        grad_mean = torch.mean(grads)
        grads = grads - grad_mean
        grads = grads / (grad_std + 1e-12)
        gradients = torch.zeros_like(inp)
        gradients[:, :3, :] = grads

    # load feature activations
    if kwargs['use_activation']:
        activations = [features['feature_{}'.format(l)] for l in layer_ids]
        losses = []
        for layer_activation in activations:
            feature_map = layer_activation[:, kwargs['channel_ids'], :]
            loss_item = torch.nn.MSELoss(reduction='mean')(feature_map, torch.zeros_like((feature_map))) #resp mean
            losses.append(loss_item)
        loss = torch.sum(torch.stack(losses))
        loss.backward()
        grads = (inp.grad)[:, :3, :].detach().clone()
        grad_std = torch.std(grads)
        grad_mean = torch.mean(grads)
        grads = grads - grad_mean
        grads = grads / (grad_std + 1e-12)
        gradients = torch.zeros_like(inp)
        gradients[:, :3, :] = grads

    # load classifier features
    if kwargs['use_clf_feats']:
        clf_feats = model.get_and_rem_clf_features()
        losses = []
        for feat in clf_feats:
            loss_item = torch.nn.MSELoss(reduction='mean')(feat, torch.zeros_like((feat)))
            losses.append(loss_item)
        loss = torch.mean(torch.stack(losses))
        loss.backward()
        grads = (inp.grad)[:, :3, :].detach().clone()  # ?
        grad_std = torch.std(grads)
        grad_mean = torch.mean(grads)
        grads = grads - grad_mean
        grads = grads / (grad_std + 1e-12)
        gradients = torch.zeros_like(inp)
        gradients[:, :3, :] = grads

    if kwargs['use_ig']:
        assert kwargs['part_ids'].shape[0] == 1 # only one part class
        baseline = integrated_gradients.get_baseline(target=inp, mode='random')
        grads = integrated_gradients.ig(F=model, x=inp, x_=baseline, n=10, part_id=kwargs['part_ids'])
        grad_std = torch.std(grads)
        grad_mean = torch.mean(grads)
        grads = grads - grad_mean
        grads = grads / (grad_std + 1e-12)
        gradients = torch.zeros_like(inp)
        gradients[:, :3, :] = grads

    if kwargs['ascent']:
        inp = inp + lr * gradients
    else:
        inp = inp - lr * gradients

    # normalize gradients
    if inp.grad is not None:
        inp.grad.zero_()

    if kwargs['normalise']:
        inp = normalise(inp, ax=0)

    if (iteration+1) % visualize_every == 0:
        pcd = o3d.geometry.PointCloud()
        pts = torch.squeeze(inp).clone().transpose(0, 1)[:, :3].detach().cpu()
        pcd.points = o3d.utility.Vector3dVector((pts.numpy()))
        o3d.visualization.draw_geometries([pcd])

    if (iteration+1) % visualize_every == 0:
        pcd_color = color_attribution(pts=torch.squeeze(orig_inp), mode='gradients', attribution=torch.squeeze(gradients, 0),
                                      color='magnitude')
        o3d.visualization.draw_geometries([pcd_color])

    return nn.Parameter(inp.clone(), requires_grad=True)


def sample_uniform_from(obj, num_samples):
    ids = np.random.choice(obj.shape[1], num_samples)
    sampled_obj = obj[:, ids]
    return sampled_obj


def load_object(path, num_samples=5000, use_scores=False, **kwargs):
    """
    5000 samples will do
    """
    obj = get_samples(path, d_type="test")
    x = sample_uniform_from(obj, num_samples=num_samples)
    x = normalise(x, ax=1)
    if use_scores:
        category = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES)
        category[cat2id(kwargs['category'])] = 1
        category = category.repeat(num_samples, 1).t()
        x = torch.vstack((x, category))

    x = nn.Parameter(torch.unsqueeze(x, 0).to(device), requires_grad=True)
    return x


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--obj_path', type=str, default="preprocessed_meshes/paired/lamp_parts_5/lamp_parts_5_style_sbd.xyz",
                   help='Path to the object')
    p.add_argument('--obj_cat', type=str, default='lamp')
    p.add_argument('--layer_ids', nargs='+', default=[0],
                   help='layer ids, any subset of [0, 1, 2, 3, 4, 5]')
    p.add_argument('--channel_ids', nargs='+', default=[0],
                   help='layer ids, any subset of [0, 1, 2, 3, 4, 5]')
    p.add_argument('--lr', type=float, default=0.01,
                   help='step size')
    p.add_argument('--num_iters', type=int, default=1,
                   help='number of iterations')
    p.add_argument('--visualize_every', type=int, default=1,
                   help='steps to visualize the modified object')
    opt = p.parse_args()

    model_path = "pretrained_models/shapenet.pointnet.pth.tar"
    obj_path = opt.obj_path
    category = opt.obj_cat
    lr = opt.lr
    iters = opt.num_iters
    visualize_every = opt.visualize_every
    channel_ids = [int(n) for n in opt.channel_ids]
    layer_ids = [int(n) for n in opt.layer_ids]

    model = load_model(model_path,
                       device=device,
                       option="segmentation",
                       new_ckpt=True,
                       global_pooling='max',
                       use_clf=True,
                       num_classes=shapenet_constants.NUM_PART_CATEGORIES,
                       num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
                       retain_feature_grad=False,
                       )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def deepdream(ch_ids, layer_ids, use_scores, use_activation, use_clf_feats, use_gbp=False):
        # load object
        x = load_object(obj_path, use_scores=use_scores, category=category).to(device) # original obj
        inp = x

        # don't modify the signatures of gradient_step()
        for n in tqdm(range(iters), desc='deepdream iteration'):
            inp = gradient_step(model=model, inp=inp, layer_ids=layer_ids, lr=lr, iteration=n,
                                use_scores=use_scores, part_ids=torch.tensor([25]), orig_inp=x,
                                use_ig=False,
                                use_activation=use_activation,
                                use_clf_feats=use_clf_feats,
                                channel_ids=ch_ids,
                                normalise=False,
                                ascent=True
                                )


    deepdream(ch_ids=torch.tensor(channel_ids), layer_ids=torch.tensor(layer_ids),
              use_scores=True, use_activation=False, use_clf_feats=False, use_gbp=False)