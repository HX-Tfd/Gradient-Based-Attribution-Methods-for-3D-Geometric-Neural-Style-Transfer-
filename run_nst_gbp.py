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



def run_grad_mask():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir
    print(nst_output_path)

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
    # chamfer_loss = chamfer_distance
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

    # hook functions needed for GBP
    forward_relu_outputs = []
    masked_features = []
    hooks = []

    def update_relus(model):
        """
            Updates all relu activation functions (backbone + classifier) so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function_features(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output (features are not to be copied)
            features = forward_relu_outputs[-1]
            corresponding_forward_output = features.clone().detach()  # copy of features
            corresponding_forward_output[corresponding_forward_output > 0] = 1  # mask of positive fwd responses

            # keep non-negative gradient responses and non-negative forward outputs
            modified_grad_out = torch.nn.Parameter(
                corresponding_forward_output.clone() * torch.clamp(grad_in[0], min=0.0))
            modified_grad_out[modified_grad_out > 0] = 1  # mask of positive bwd responses
            masked_features.append(features * modified_grad_out)
            masked_features[-1].requires_grad = True

            del forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass (features)
            """
            forward_relu_outputs.append(ten_out)

        # only hook the backbone
        for _, module in model._modules.items():
            for l in module.modules():
                if isinstance(l, nn.ReLU):
                    hooks.append(l.register_full_backward_hook(relu_backward_hook_function_features))
                    hooks.append(l.register_forward_hook(relu_forward_hook_function))

    update_relus(encoder)

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
        inp_obj = torch.unsqueeze(torch.vstack
                                  ((obj, category_c)), 0).to(device)
        E_O_tilde, scores = encoder(inp_obj)
        parts_loss = mse_loss(scores_Os[0, opt.part_id_s, :], scores[0, opt.part_id_c, :])
        print("parts loss: ", parts_loss)
        parts_loss.backward()
        print("obj grad: ", obj.grad)
        obj.grad.zero_()
        masked_stylized_features = masked_features
        stylised_feats = (masked_stylized_features[::-1])[:5]
        stylised_feats.append(E_O_tilde["feature_5"]) # global feature

        for j in range(len(stylised_feats)):
            print(stylised_feats[j].shape)
            if j in opt.content_layers:
                stylised_feats_c.append(stylised_feats[j])
            if j in opt.style_layers:
                stylised_feats_s.append(stylised_feats[j])


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
        masked_features = [] # for next iter.
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
    opt.mode = 'gbp'  # set attribution mode, currently the only implemented attribution-mask method
    opt.part_id_c = 25  # provide content part id
    opt.part_id_s = 47  # provide style part id
    run_grad_mask()

    # lc = []
    # opt.summarize = False
    # for i in range(6):
    #     lc.append(i)
    #     opt.output_dir = "figures/opt_progressive_content_layers_only/{}".format(i)
    #     opt.content_layers = lc
    #     opt.content_weight = 1
    #     opt.style_weight = 0
    #     run_grad_mask()

