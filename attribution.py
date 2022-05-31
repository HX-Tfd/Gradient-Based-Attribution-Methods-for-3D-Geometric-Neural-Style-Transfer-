import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm

from guided_bp import GuidedBackprop
from utils import *
import shapenet_constants
import integrated_gradients
from modules.pointnet import PointNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Gradient attribution methods testing ground
"""


def load_object(path, num_samples=1024, use_scores=False, **kwargs):
    """
    5000 samples will do, more samples does not give that much difference
    """
    obj = get_samples(path=path, d_type='txt', with_normalization=True, to_tensor=True)
    x = sample_uniform_from(obj, num_samples=num_samples, replace=True)
    if use_scores:
        category = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES)
        category[cat2id(kwargs['category'])] = 1
        category = category.repeat(num_samples, 1).t()
        x = torch.vstack((x, category))

    x = nn.Parameter(torch.unsqueeze(x, 0).clone(), requires_grad=True).to(device)
    return x


def load_seg_model(model_path):
    model = load_model(model_path,
                       device=device,
                       option="segmentation",
                       new_ckpt=True,
                       global_pooling='max',
                       use_clf=True,
                       num_classes=shapenet_constants.NUM_PART_CATEGORIES,
                       num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
                       retain_feature_grad=True,
                       )
    return model


def load_clf_model(model_path):
    model = load_model(path=model_path, device=device, option='classification', k=len(get_shapename2id()),
                       global_pooling='max',
                       use_clf=True,
                       new_ckpt=False,
                       retain_feature_grad=True
                       )
    return model

def test_GradCAM(model, obj, mode, layer_id=None, part_id=None, cat=None):
    from GradCAM import gradCAM_clf, gradCAM_seg
    grads = None
    layer_id = layer_id if layer_id is not None else 7 # last layer before last conv
    if mode == 'clf':
        grads = gradCAM_clf(model, layer_id=layer_id, inp=obj, cat_id=name2id[cat.upper()]) # airplane
    elif mode == 'seg':
        grads = gradCAM_seg(model, layer_id=layer_id, inp=obj, part_class=part_id)
    return grads


def test_InterGrad(model, obj, mode, num_steps=50, part_id=None, cat=None):
    # TESTING Integrated Gradients
    baseline = integrated_gradients.get_baseline(target=obj, mode='zero')
    grads = None
    if mode == 'clf':
        grads = integrated_gradients.ig_clf(F=model, x=obj, x_=baseline, n=num_steps,
                                        cat_id=name2id[cat.upper()])[:, :3, :]
    elif mode == 'seg':
        grads = integrated_gradients.ig_seg(F=model, x=obj, x_=baseline, n=num_steps,
                                        part_id=part_id, elem_wise=True)[:, :3, :]
    return grads


def test_GBP(model, obj, num_samples):
    # TESTING Guided Backprop
    target = torch.zeros(1, 10, num_samples)
    GBP = GuidedBackprop(model, eval=False, hook=True, hook_first=False)
    GBP.generate_gradients_parts(obj, None, id=torch.tensor([25]))
    grads, inter_grads = GBP.get_gradients()


def test_attr_clf():
    """
    Not used
    """
    use_scores = True
    num_samples = 512
    obj_cat = 'chair'

    mode = 'grad_cam'
    model = load_clf_model(model_path="pretrained_models/clf_ep_24_maj.pth")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    obj_path = "data/PartAnnotation/03636649/points/732a2b4f7626ca61d197f67767b32741.pts"
    obj = get_samples(path=obj_path, d_type='txt', with_normalization=False, to_tensor=True)
    obj = sample_uniform_from(obj, num_samples=num_samples, replace=True)
    # obj = load_object(path=obj_path, use_scores=use_scores, category=obj_cat, num_samples=num_samples)
    obj.requires_grad = True

    if mode == 'grad_cam':
        grads = test_GradCAM(model, torch.unsqueeze(obj, 0), cat=obj_cat)
        grads = standardize(grads)
        grads[grads < 0] = 0
        pcd = color_attribution(pts=obj, mode='gradients', attribution=torch.squeeze(grads, 0),  # default gradients
                                color='negpos')
        o3d.visualization.draw_geometries([pcd])

    elif mode == 'gbp':
        class_id = 15
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                model.register_gbp_hooks(m)
        scores = model(torch.unsqueeze(obj, 0))['scores']
        scores[:, class_id].backward()
        grads = obj.grad  # [1, 3, N]
        pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=torch.squeeze(grads, 0),
                                # default gradients
                                color='negpos')
        o3d.visualization.draw_geometries([pcd])

    elif mode == 'inter_grad':
        grads = test_InterGrad(torch.unsqueeze(obj, 0), mode='clf', cat=obj_cat)
        pcd = color_attribution(pts=obj, mode='gradients', attribution=torch.squeeze(grads, 0),
                                color='magnitude')
        o3d.visualization.draw_geometries([pcd])


def test_attr_seg(p):
    num_samples = p.num_samples
    pretrained = opt.pretrained

    if pretrained:
        model = load_seg_model(model_path = "pretrained_models/shapenet.pointnet.pth.tar")
    else:
        model = PointNet(
            num_classes=shapenet_constants.NUM_PART_CATEGORIES,
            num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
            retain_feature_grad=False if opt.mode == 'grad_cam' else True
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    obj_path = p.obj_path
    obj_cat = p.obj_cat
    print(get_id2part(obj_cat))
    obj = load_object(path=obj_path, use_scores=True, category=obj_cat, num_samples=num_samples)

    if p.color_samples:
        color_samples(obj, model, visualize=True)

    if p.mode == 'grads':
        class_id = p.part_id  # define here
        _, scores = model(obj)
        grads = []
        for n in tqdm(range(num_samples)):
            scores[:, class_id, n].backward(retain_graph=True)
            grad = obj.grad.clone()
            grads.append(grad)
        grads = sum(grads)/len(grads)
        if p.vis_negpos:
            pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=torch.squeeze(grads, 0),
                                    color='negpos')
            o3d.visualization.draw_geometries([pcd])
        if p.vis_magnitude:
            pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=torch.squeeze(grads, 0),
                                    color='magnitude')
            o3d.visualization.draw_geometries([pcd])

    elif p.mode == 'grad_cam':
        class_id = p.part_id
        lid = p.layer_id
        grads, _, _ = test_GradCAM(model, obj, mode='seg', part_id=class_id, layer_id=lid)
        grads = norm_unit(grads)
        if p.vis_negpos:
            pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=torch.squeeze(grads, 0),  # default gradients
                                    color='negpos')
            o3d.visualization.draw_geometries([pcd])
        # if p.vis_magnitude:
        #     pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=grads,  # default gradients
        #                             color='magnitude')
        #     o3d.visualization.draw_geometries([pcd])

    elif p.mode == 'gbp':
        class_id = p.part_id # define here
        obj.retain_grad()

        for m in model.modules():
            if isinstance(m, nn.ReLU):
                model.register_gbp_hooks(m)
        feats, scores = model(obj)

        g = [] # object grads
        fwd_feats = model.get_gbp_fwd_features(remove=False, clone=True)

        for i in tqdm(range(num_samples)):
            model.forward_relu_outputs = fwd_feats.copy()
            pt_score = scores[:, class_id, i]
            pt_score.backward(retain_graph=True)
            grad = obj.grad.clone()
            model.bwd_feature_grads = []
            grad = torch.mean(grad, dim=1)#max(grad, dim=1).values
            if p.gbp_mask_negative:
                grad[grad<0] = 0
            g.append(grad)  # [1, 3, N]
            obj.grad.zero_()

        grads = sum(g)/num_samples
        grads = norm_unit(grads)

        if p.vis_negpos:
            pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=grads,
                                    color='negpos')
            o3d.visualization.draw_geometries([pcd])
        if p.vis_magnitude:
            pcd = color_attribution(pts=torch.squeeze(obj), mode='gradients', attribution=grads,
                                    color='magnitude')
            o3d.visualization.draw_geometries([pcd])

    elif p.mode == 'ig':
        obj.retain_grad()
        part_id = opt.part_id
        n_steps = p.n_steps
        grads = test_InterGrad(model=model, obj=obj, mode='seg', part_id=part_id, num_steps=n_steps)
        if p.vis_negpos:
            pcd = color_attribution(pts=torch.squeeze(obj, 0), mode='gradients', attribution=torch.squeeze(grads, 0),
                                    color='negpos')
            o3d.visualization.draw_geometries([pcd])
        if p.vis_magnitude:
            pcd = color_attribution(pts=torch.squeeze(obj, 0), mode='gradients', attribution=torch.squeeze(grads, 0),
                                    color='magnitude')
            o3d.visualization.draw_geometries([pcd])

    else:
        print("unsupported mode: ", opt.mode)
        print("mode must be one of [gbp, ig, grad_cam]")


if __name__ == '__main__':
    name2id = {name: id for id, name in enumerate(list(get_shapename2id().keys()), 0)}
    print(name2id)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--obj_path', type=str, default="data/PartAnnotation/03636649/points/1b9b6f1ddf363e0e8b424343280aeccb.pts",
                   help='Path to the object')
    p.add_argument('--obj_cat', type=str, default='lamp',
                   help='Category of the object')
    p.add_argument('--num_samples', type=int, default='512')
    p.add_argument('--part_id', type=int, required=True,
                   help='part id corresponding to the object')
    p.add_argument('--vis_negpos', action='store_true',
                   help='visualize using sum of gradient values along the coordinate dimension'
                   )
    p.add_argument('--vis_magnitude', action='store_true',
                   help='visualize using the magnitude of the gradient along the coordinate dimension'
                   )
    p.add_argument('--mode', type=str, default='grad_cam',
                   help='type of the attribution method, one of [gbp, ig, grad_cam]')
    p.add_argument('--color_samples', action='store_true',
                   help='visualize the part segmentation')
    p.add_argument('--pretrained', action='store_true',
                   help='whether to use a pretrained model')
    # gbp
    p.add_argument('--gbp_mask_negative', action='store_true',
                   help='whether to apply ReLU on the gradients')

    # ig
    p.add_argument('--n_steps', type=int, default=10,
                   help='number of discrete timesteps')

    # gradcam/hirescam
    p.add_argument('--layer_id', type=int, default=7,
                   help='id of the convolutional layer used to compute GradCAM/HiResCAM scores')
    opt = p.parse_args()

    test_attr_seg(opt)