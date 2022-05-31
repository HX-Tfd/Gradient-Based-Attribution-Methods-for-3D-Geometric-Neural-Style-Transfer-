import time
import seaborn as sns

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import *
from modules import loss

from guided_bp import GuidedBackprop
from GradCAM import gradCAM_seg
import shapenet_constants

import open3d as o3d


"""
global configs
"""
arg_parser = False # one of [argparse, exp_config]
device = "cuda" if torch.cuda.is_available() else "cpu"
if arg_parser: # deprecated
    opt = ExperimentParser().opt.parse_args()
else:
    from configs import nst_opt_based as opt

"""
Experiments
"""

def run_nst_grad_cam():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir

    # load data
    print("loading NST data ...")
    O_c = get_samples(path=opt.content_shape, d_type='txt', with_normalization=True, to_tensor=True)
    O_s = get_samples(path=opt.style_shape, d_type='txt', with_normalization=True, to_tensor=True)

    encoder = load_model(encoder_path,
                         device=device,
                         option="segmentation",
                         new_ckpt=True,
                         global_pooling='max',
                         use_clf=True,
                         num_classes=shapenet_constants.NUM_PART_CATEGORIES,
                         num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
                         retain_feature_grad=False,
                         )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # sample same number of points and normalise into [-1, 1]
    num_samples = opt.num_points
    print("using %d samples" % num_samples)
    O_c_ = sample_uniform_from(O_c, num_samples, replace=True)
    O_s_ = sample_uniform_from(O_s, num_samples, replace=True)

    # append shape category (one-hot vecs)
    category_s, category_c = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES), torch.zeros(
        shapenet_constants.NUM_SHAPE_CATEGORIES)
    category_c[cat2id(opt.content_category)] = 1
    category_s[cat2id(opt.style_category)] = 1
    category_c = category_c.repeat(num_samples, 1).t()
    category_s = category_s.repeat(num_samples, 1).t()
    if opt.normalize:
        print("(re-)normalizing input points ...")
        O_c_, O_s_ = normalise(O_c_, ax=1), normalise(O_s_, ax=1)
    O_c = torch.vstack((O_c_, category_c))  # only for torch >= 1.8.0 O_c_ are the coords!!!!!
    O_s = torch.vstack((O_s_, category_s))
    O_c = nn.Parameter(torch.unsqueeze(O_c, 0).to(device), requires_grad=True)
    O_s = nn.Parameter(torch.unsqueeze(O_s, 0).to(device), requires_grad=True)

    # attribution loss initialization
    part_class_c, part_class_s = opt.content_part_class, opt.style_part_class
    grads_c, E_Oc, scores_Oc = gradCAM_seg(encoder, layer_id=7, inp=O_c, part_class=part_class_c)
    grads_s, E_Os, scores_Os = gradCAM_seg(encoder, layer_id=7, inp=O_s, part_class=part_class_s)
    content_scores, target_scores = norm_unit(grads_c), norm_unit(grads_s)

    # layers config
    '''
    feature_0...4: intermediate features
    feature_5: global feature

    features are of size [B, S, num_points], S \in [64, 128, 512, 2048]
    '''
    content_feats, style_feats = [], []

    for k, v in E_Oc.items():
        if int(k[8]) in opt.content_layers:
            content_feats.append(v.detach() * content_scores) #[1, C, N], [1, N]
    for k, v in E_Os.items():
        if int(k[8]) in opt.style_layers:
            style_feats.append(v.detach() * target_scores)


    params_to_opt = []
    fixed_params = []
    if opt.part_specific:
        # initialize object, fix content parts, dict
        mask_class_ids = opt.mask_class_list
        pred_parts = scores_Oc.max(dim=1).indices.squeeze(0)

        # convert params from dict to tensor again without cat_c
        for i in range(len(pred_parts)):
            if pred_parts[i] not in mask_class_ids:
                fixed_params.append(O_c_[:, i].clone())
            else:
                params_to_opt.append(nn.Parameter(O_c_[:, i].clone(), requires_grad=True))
    else:
        params_to_opt = [nn.Parameter(O_c_.clone(), requires_grad=True)]

    # training configs
    exp_name = opt.layers_exp_suffix
    num_iters = opt.num_iters
    step_size = opt.step_size

    optimizer = torch.optim.Adam(params=params_to_opt, lr=step_size)

    mse_loss = nn.MSELoss().to(device)
    # chamfer_loss = chamfer_distance
    content_weight = opt.content_weight
    style_loss = loss.StatsLoss().to(device)
    style_weight = opt.style_weight
    save_every = opt.save_every
    writer = SummaryWriter(log_dir=nst_output_path, filename_suffix=exp_name)

    # initialise running scalars for tensorboard
    verbose = opt.verbose
    summarize = opt.summarize
    min_total_loss = torch.tensor(2 ** 31.)
    min_weighted_style_loss = torch.tensor(2 ** 31.)
    opt_iter = 0
    prev_total_loss = torch.tensor(2 ** 31)

    print('Starting NST')
    start = time.time()
    for n in tqdm(range(num_iters)):
        # assemble object
        if opt.part_specific:
            coords = fixed_params + params_to_opt
            obj = torch.zeros_like(O_c_)
            for i in range(O_c_.shape[1]):
                obj[:, i] = coords[i]
        else:
            obj = params_to_opt[0]

        # training
        optimizer.zero_grad()
        stylised_feats_c, stylised_feats_s = [], []
        content_losses, style_losses = [], []

        inp_obj = torch.unsqueeze(torch.vstack((obj, category_c)), 0).to(device)
        E_O_tilde, _ = encoder(inp_obj)

        grads, _, _ = gradCAM_seg(encoder, layer_id=7, inp=inp_obj, part_class=part_class_c)
        grads = norm_unit(grads)
        if opt.style_only:
           attr_loss = mse_loss(grads, target_scores)
        elif opt.content_only:
            attr_loss = mse_loss(grads, content_scores)
        else:
            attr_loss = mse_loss(grads, target_scores) + mse_loss(grads, content_scores)

        for k, v in E_O_tilde.items():
            lid = int(k[8])
            if lid in opt.content_layers:
                # print("computing layer {} content features".format(lid))
                # grads, _, _ = gradCAM_seg(encoder, layer_id=lid, inp=inp_obj, part_class=part_class_c)
                # content_scores = norm_unit(grads) #?
                # stylised_feats_c.append(v * content_scores)
                stylised_feats_c.append(v)
            if lid in opt.style_layers:
                # print("computing layer {} style features".format(lid))
                # grads, _, _ = gradCAM_seg(encoder, layer_id=lid, inp=inp_obj, part_class=part_class_c)
                # style_scores = norm_unit(grads)
                # stylised_feats_s.append(v * style_scores)
                stylised_feats_s.append(v)

        for i in range(len(stylised_feats_c)):
            cf = content_feats[i]
            styf = stylised_feats_c[i]
            content_losses.append(mse_loss(cf, styf))

        for i in range(len(stylised_feats_s)):
            sf = style_feats[i]
            styf = stylised_feats_s[i]
            style_losses.append(
                style_loss(sf, styf, dim=0 if opt.style_layers[i] == 5 else 1))  # ?? uncessary to distinguish layers?

        cl = sum(content_losses)
        sl = sum(style_losses)

        if opt.full_loss:
            total_loss = content_weight * cl + style_weight * sl + opt.attr_weight * attr_loss
        else:
            total_loss = opt.attr_weight * attr_loss

        # training stats
        if summarize:
            prev_total_loss = total_loss
            if total_loss < min_total_loss:
                opt_iter = n
            min_weighted_style_loss = min(min_weighted_style_loss, style_weight * sl)
            min_total_loss = min(min_total_loss, total_loss)

            writer.add_scalar("{}/content loss".format(exp_name), cl, n)
            writer.add_scalars("{}/weighted style loss (with min)".format(exp_name),
                               {'min_weighted_style_loss': min_weighted_style_loss,
                                'weighted_style_loss': style_weight * sl},
                               n)
            writer.add_scalars("{}/total loss (with min)".format(exp_name),
                               {'min_total_loss': min_total_loss,
                                'total_loss': total_loss},
                               n)
            writer.add_scalar("{}/weighted attribution loss".format(exp_name), opt.attr_weight * attr_loss, n)
            writer.add_scalar("{}/optimal iter for total loss".format(exp_name), opt_iter, n)
        if verbose:
            print("iter {}, unweighted attr loss:{},  total loss: {}".format(n, attr_loss, total_loss))

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if n % save_every == 0 and n > 0:
            save_to(torch.squeeze(obj).transpose(0, 1).detach(), n, path=nst_output_path, format='ply',
                    with_color=False)
        obj = nn.Parameter(obj.clone(), requires_grad=True)
    end = time.time()
    print('training time: ', end - start)

    # save object
    if opt.with_color:
        num_classes, palette_size = 50, 6
        _, scores = encoder(torch.unsqueeze(torch.vstack((obj, category_c)), 0).to(device))  # 1, 50, 32000
        part_classes = torch.max(torch.squeeze(scores), dim=0).indices
        colors = torch.zeros(num_samples, 3)
        col_list = sns.color_palette("husl", palette_size)
        for i in range(colors.shape[0]):
            colors[i, :] = torch.tensor(col_list[part_classes[i] % palette_size])
        obj = torch.hstack((torch.squeeze(obj).transpose(0, 1).detach(), colors))
        save_to(obj,
                "_final_{}".format(exp_name),
                path=nst_output_path,
                with_color=True)
    else:
        save_to(torch.squeeze(obj).transpose(0, 1).detach(),
                "_final_{}".format(exp_name),
                path=nst_output_path,
                format='ply',
                with_color=False)


def run_nst_part_specific():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir

    # load data
    print("loading NST data ...")
    # the directory that holds the pre-processed (sampled) points
    # O_c = get_samples(opt.content_shape, d_type="content")
    # O_s = get_samples(opt.style_shape, d_type="style")
    # optional standardization?
    O_c = get_samples(path=opt.content_shape, d_type='txt', with_normalization=True, to_tensor=True)
    O_s = get_samples(path=opt.style_shape, d_type='txt', with_normalization=True, to_tensor=True)

    encoder = load_model(encoder_path,
                       device=device,
                       option="segmentation",
                       new_ckpt=True,
                       global_pooling='max',
                       use_clf=True,
                       num_classes=shapenet_constants.NUM_PART_CATEGORIES,
                       num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
                       retain_feature_grad=False,
                       )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # sample same number of points and normalise into [-1, 1]
    num_samples = opt.num_points
    print("using %d samples" % num_samples)
    O_c_ = sample_uniform_from(O_c, num_samples, replace=True)
    O_s_ = sample_uniform_from(O_s, num_samples, replace=True)

    # append shape category (one-hot vecs)
    category_s, category_c = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES), torch.zeros(
        shapenet_constants.NUM_SHAPE_CATEGORIES)
    category_c[cat2id(opt.content_category)] = 1
    category_s[cat2id(opt.style_category)] = 1
    category_c = category_c.repeat(num_samples, 1).t()
    category_s = category_s.repeat(num_samples, 1).t()
    if opt.normalize:
        print("(re-)normalizing input points ...")
        O_c_, O_s_ = normalise(O_c_, ax=1), normalise(O_s_, ax=1)
    O_c = torch.vstack((O_c_, category_c))  # only for torch >= 1.8.0 O_c_ are the coords!!!!!
    O_s = torch.vstack((O_s_, category_s))

    O_c = torch.unsqueeze(O_c, 0).to(device)
    O_s = torch.unsqueeze(O_s, 0).to(device)

    E_Oc, scores_Oc = encoder(O_c)
    E_Os, scores_Os = encoder(O_s)  # [1, f, N]

    # layers config
    content_feats, style_feats = [], []

    for k, v in E_Oc.items():
        if int(k[8]) in opt.content_layers:
            content_feats.append(v.detach())
    for k, v in E_Os.items():
        if int(k[8]) in opt.style_layers:
            style_feats.append(v.detach())

    # initialize object, fix parts, dict
    mask_class_ids = opt.mask_class_list
    pred_parts = scores_Oc.max(dim=1).indices.squeeze(0)

    # convert params from dict to tensor again without cat_c
    fixed_params, fixed_ids = [], []
    params_to_opt = []
    for i in range(len(pred_parts)):
        if pred_parts[i] not in mask_class_ids:
            fixed_params.append(O_c_[:, i].clone())
            fixed_ids.append(i)
        else:
            params_to_opt.append(nn.Parameter(O_c_[:, i].clone(), requires_grad=True))

    # training configs
    exp_name = opt.layers_exp_suffix
    num_iters = opt.num_iters
    step_size = opt.step_size

    optimizer = torch.optim.Adam(params=params_to_opt, lr=step_size)

    mse_loss = nn.MSELoss().to(device)
    # chamfer_loss = chamfer_distance
    content_weight = opt.content_weight
    style_loss = loss.StatsLoss().to(device)
    style_weight = opt.style_weight
    save_every = opt.save_every
    writer = SummaryWriter(log_dir=nst_output_path, filename_suffix=exp_name)

    # initialise running scalars for tensorboard
    verbose = opt.verbose
    summarize = opt.summarize
    min_total_loss = torch.tensor(2 ** 31.)
    min_weighted_style_loss = torch.tensor(2 ** 31.)
    opt_iter = 0
    prev_total_loss = torch.tensor(2 ** 31)

    print('Starting NST')
    start = time.time()
    for n in tqdm(range(num_iters)):
        # assemble object
        obj = torch.zeros_like(O_c_)
        coords = fixed_params + params_to_opt
        for i in range(O_c_.shape[1]):
            obj[:, i] = coords[i]

        # training
        optimizer.zero_grad()
        stylised_feats_c, stylised_feats_s = [], []
        content_losses, style_losses = [], []
        parts_loss = 0
        inp_obj = torch.unsqueeze(torch.vstack((obj, category_c)), 0).to(device)
        E_O_tilde, scores = encoder(inp_obj)

        for k, v in E_O_tilde.items():
            if int(k[8]) in opt.content_layers:
                stylised_feats_c.append(v)
            if int(k[8]) in opt.style_layers:
                stylised_feats_s.append(v)

        for i in range(len(stylised_feats_c)):
            cf = content_feats[i]
            styf = stylised_feats_c[i]
            content_losses.append(mse_loss(cf, styf))

        for i in range(len(stylised_feats_s)):
            sf = style_feats[i]
            styf = stylised_feats_s[i]
            style_losses.append(
                style_loss(sf, styf, dim=0 if opt.style_layers[i] == 5 else 1))  # ?? uncessary to distinguish layers?

        cl = sum(content_losses)
        sl = sum(style_losses)
        score_loss = mse_loss(scores[:, opt.content_part_class, fixed_ids],
                              scores_Os[:, opt.style_part_class, fixed_ids])
        score_weight = 10
        total_loss = content_weight * cl + style_weight * sl + score_weight * score_loss

        # training stats
        if summarize:
            prev_total_loss = total_loss
            if total_loss < min_total_loss:
                opt_iter = n
            min_weighted_style_loss = min(min_weighted_style_loss, style_weight * sl)
            min_total_loss = min(min_total_loss, total_loss)

            writer.add_scalar("{}/content loss".format(exp_name), cl, n)
            writer.add_scalars("{}/weighted style loss (with min)".format(exp_name),
                               {'min_weighted_style_loss': min_weighted_style_loss,
                                'weighted_style_loss': style_weight * sl},
                               n)
            writer.add_scalars("{}/total loss (with min)".format(exp_name),
                               {'min_total_loss': min_total_loss,
                                'total_loss': total_loss},
                               n)
            writer.add_scalar("{}/weighted score loss".format(exp_name), score_weight * score_loss, n)
            writer.add_scalar("{}/optimal iter for total loss".format(exp_name), opt_iter, n)

        if verbose:
            print("iter {}, total loss: {}".format(n, total_loss))

        total_loss.backward()
        optimizer.step()
        if n % save_every == 0 and n > 0:
            save_to(torch.squeeze(obj).transpose(0, 1).detach(), n, path=nst_output_path, format='ply',
                    with_color=False)
        obj = nn.Parameter(obj.clone(), requires_grad=True)
    end = time.time()
    print('training time: ', end - start)
    print("============================")

    # save object
    if opt.with_color:
        num_classes, palette_size = 50, 6
        _, scores = encoder(torch.unsqueeze(torch.vstack((obj, category_c)), 0).to(device))  # 1, 50, 32000
        part_classes = torch.max(torch.squeeze(scores), dim=0).indices
        colors = torch.zeros(num_samples, 3)
        col_list = sns.color_palette("husl", palette_size)
        for i in range(colors.shape[0]):
            colors[i, :] = torch.tensor(col_list[part_classes[i] % palette_size])
        obj = torch.hstack((torch.squeeze(obj).transpose(0, 1).detach(), colors))
        save_to(obj,
                "_final_{}".format(exp_name),
                path=nst_output_path,
                with_color=True)
    else:
        save_to(torch.squeeze(obj).transpose(0, 1).detach(),
                "_final_{}".format(exp_name),
                path=nst_output_path,
                format='ply',
                with_color=False)



def run_grad_mask():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir
    print(nst_output_path)

    # load data
    print("loading NST data ...")
    # the directory that holds the pre-processed (sampled) points
    # O_c = get_samples(opt.content_shape, d_type="content")
    # O_s = get_samples(opt.style_shape, d_type="style")
    # optional standardization?
    O_c = get_samples(path=opt.content_shape, d_type='txt', with_normalization=True, to_tensor=True)
    O_s = get_samples(path=opt.style_shape, d_type='txt', with_normalization=True, to_tensor=True)

    encoder = load_model(encoder_path,
                       device=device,
                       option="segmentation",
                       new_ckpt=True,
                       global_pooling='max',
                       use_clf=True,
                       num_classes=shapenet_constants.NUM_PART_CATEGORIES,
                       num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
                       retain_feature_grad=False,
                       )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # sample same number of points and normalise into [-1, 1]
    num_samples = opt.num_points
    print("using %d samples" % num_samples)
    O_c_ = sample_uniform_from(O_c, num_samples, replace=True)
    O_s_ = sample_uniform_from(O_s, num_samples, replace=True)

    # append shape category (one-hot vecs)
    category_s, category_c = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES), torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES)
    category_c[cat2id(opt.content_category)] = 1
    category_s[cat2id(opt.style_category)] = 1
    category_c = category_c.repeat(num_samples, 1).t()
    category_s = category_s.repeat(num_samples, 1).t()
    if opt.normalize:
        print("(re-)normalizing input points ...")
        O_c_, O_s_ = normalise(O_c_, ax=1), normalise(O_s_, ax=1)
    O_c = torch.vstack((O_c_, category_c)) # only for torch >= 1.8.0 O_c_ are the coords!!!!!
    O_s = torch.vstack((O_s_, category_s))

    O_c = torch.unsqueeze(O_c, 0).to(device)
    O_s = torch.unsqueeze(O_s, 0).to(device)
    E_Oc, scores_Oc = encoder(O_c)
    E_Os, scores_Os = encoder(O_s)  # [1, f, N]

    # layers config
    content_feats, style_feats = [], []

    for k, v in E_Oc.items():
        if int(k[8]) in opt.content_layers:
            content_feats.append(v.detach())
    for k, v in E_Os.items():
        if int(k[8]) in opt.style_layers:
            style_feats.append(v.detach())

    # initialize object, fix parts
    fixed_class_id = 25
    obj_ = O_c_.clone()
    inp_obj = nn.Parameter(torch.unsqueeze(torch.vstack((obj_, category_c)), 0), requires_grad=True).to(device)
    _, scores = encoder(inp_obj)

    if opt.randn_noise:
        obj = nn.Parameter(torch.randn_like(obj_).to(device), requires_grad=True)
    else:
        obj = nn.Parameter(obj_.to(device), requires_grad=True)

    # training configs
    exp_name = opt.layers_exp_suffix
    num_iters = opt.num_iters
    step_size = opt.step_size

    optimizer = torch.optim.Adam(params=[obj], lr=step_size)

    mse_loss = nn.MSELoss().to(device)
    content_weight = opt.content_weight
    style_loss = loss.StatsLoss().to(device)
    style_weight = opt.style_weight
    print("style weight: ", style_weight)
    print("output path: ", nst_output_path)
    save_every = opt.save_every
    writer = SummaryWriter(log_dir=nst_output_path, filename_suffix=exp_name)

    # initialise running scalars for tensorboard
    verbose = opt.verbose
    summarize = opt.summarize
    min_total_loss = torch.tensor(2 ** 31.)
    min_weighted_style_loss = torch.tensor(2 ** 31.)
    opt_iter = 0
    prev_total_loss = torch.tensor(2 ** 31)

    print('Starting NST')
    start = time.time()

    """
    Two passes: first pass computes the masked features, second pass uses the masked features for nst training
    """
    for n in tqdm(range(num_iters)):
        encoder.zero_grad()
        if n % save_every == 0 and n > 0:
            save_to(torch.squeeze(obj).transpose(0, 1).detach(), n, path=nst_output_path, format='ply',
                    with_color=False)

        stylised_feats_c, stylised_feats_s = [], []
        content_losses, style_losses = [], []
        parts_loss = 0
        inp_obj = torch.unsqueeze(torch.vstack
                                  ((obj, category_c)), 0).to(device)
        E_O_tilde, scores = encoder(inp_obj)

        if opt.mode == 'gbp':
            assert opt.part_id_c is not None and opt.part_id_s is not None
            # initialise Guided Backpropagation
            GBP = GuidedBackprop(model=encoder, eval=False, hook=True, hook_first=False)

            parts_loss = GBP.generate_gradients_parts(inp_obj,
                                                      target_part_scores=scores_Os,
                                                      content_part_id=opt.part_id_c,
                                                      custom=opt.gbp_mode,
                                                      id=opt.part_id_s)
            masked_stylized_features = GBP.masked_features
            stylised_feats = (masked_stylized_features[::-1])[:5]  # if features of clf not used
            stylised_feats.append(E_O_tilde["feature_5"])  # add global feature
            for j in range(len(stylised_feats)):
                print(stylised_feats[j].shape)
                if j in opt.content_layers:
                    stylised_feats_c.append(stylised_feats[j])
                if j in opt.style_layers:
                    stylised_feats_s.append(stylised_feats[j])

        elif opt.mode == 'int_grad': # not implemented
            pass

        else:
            E_O_tilde, scores = encoder(inp_obj)
            for k, v in E_O_tilde.items():
                if int(k[8]) in opt.content_layers:
                    stylised_feats_c.append(v)
                if int(k[8]) in opt.style_layers:
                    stylised_feats_s.append(v)

        for i in range(len(stylised_feats_c)):
            cf = content_feats[i]
            styf = stylised_feats_c[i]
            content_losses.append(mse_loss(cf, styf))

        for i in range(len(stylised_feats_s)):
            sf = style_feats[i]
            styf = stylised_feats_s[i]
            style_losses.append(
                style_loss(sf, styf, dim=0 if opt.style_layers[i] == 5 else 1))  # ?? uncessary to distinguish layers?

        cl = sum(content_losses)
        sl = sum(style_losses)
        total_loss = content_weight * cl + style_weight * sl

        # training stats
        if summarize:
            prev_total_loss = total_loss
            if total_loss < min_total_loss:
                opt_iter = n
            min_weighted_style_loss = min(min_weighted_style_loss, style_weight * sl)
            min_total_loss = min(min_total_loss, total_loss)

            writer.add_scalar("{}/content loss".format(exp_name), cl, n)
            writer.add_scalars("{}/weighted style loss (with min)".format(exp_name),
                               {'min_weighted_style_loss': min_weighted_style_loss,
                                'weighted_style_loss': style_weight * sl},
                               n)
            writer.add_scalars("{}/total loss (with min)".format(exp_name),
                               {'min_total_loss': min_total_loss,
                                'total_loss': total_loss},
                               n)
            writer.add_scalar("{}/parts loss".format(exp_name), parts_loss, n)
            writer.add_scalar("{}/optimal iter for total loss".format(exp_name), opt_iter, n)

        if verbose:
            print("iter {}, total loss: {}".format(n, total_loss))
        optimizer.zero_grad()
        total_loss.backward()
        print("input grad: ", obj.grad)
        optimizer.step()
    end = time.time()
    print('training time: ', end - start)

    # save object
    if opt.with_color:
        num_classes, palette_size = 50, 6
        _, scores = encoder(torch.unsqueeze(torch.vstack((obj, category_c)), 0).to(device)) #1, 50, 32000
        part_classes = torch.max(torch.squeeze(scores), dim=0).indices
        colors = torch.zeros(num_samples, 3)
        col_list = sns.color_palette("husl", palette_size)
        for i in range(colors.shape[0]):
            colors[i, :] = torch.tensor(col_list[part_classes[i] % palette_size])
        obj = torch.hstack((torch.squeeze(obj).transpose(0, 1).detach(), colors))
        save_to(obj,
                "_final_{}".format(exp_name),
                path=nst_output_path,
                with_color=True)
    else:
        save_to(torch.squeeze(obj).transpose(0, 1).detach(),
                "_final_{}".format(exp_name),
                path=nst_output_path,
                format='ply',
                with_color=False)

"""
Define experiments here
"""
if __name__ == '__main__':

    """
    run experiment
    example
    """
    style_shape = 'data/PartAnnotation/04379243/points/1ac080a115a94477c9fc9da372dd139a.pts'
    style_category = 'table'
    content_shape = 'data/PartAnnotation/03001627/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts'
    content_category = 'chair'
    opt.mask_class_list = [13]  # these parts of the content object will be optimized
    opt.content_part_class = 13  # part id of the content object to be optimized
    opt.style_part_class = 47  # transferring to the style of this target part
    run_nst_part_specific()

