import time
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import open3d as o3d

from modules import loss
from modules.pointnet import PointNet
from configs import nst_opt_based as opt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path, option='encoder'):
    print("loading {} model".format(option))
    if option == 'encoder':
        model = PointNet(
            num_classes=50,
            num_shapes=16,
            extra_feature_channels=0, # pointnet takes Nx3 shape input by default
            global_pooling='avg'
        )
    else:
        raise RuntimeError("unidentified option: ", option)

    checkpoint = torch.load(path, map_location=torch.device(device))
    new_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        new_dict[name] = v
    model.load_state_dict(new_dict)
    print("load state dict done")
    return model


def get_samples(path, d_type, o_type="pcd", normalise=False):
    print(o_type)
    if o_type == "pcd":
        print("reading {} points ...".format(o_type))
        pcd = o3d.io.read_point_cloud(path)
        vertices = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if normalise:
            normalise(normals)
        features = np.hstack((vertices, normals))
        print("--> {} {} points in total".format(vertices.shape[0], o_type))
    else:
        print("reading {} vertices ...".format(o_type))
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = np.asarray(mesh.vertices)
         # color or normal
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.triangle_normals)
        if normalise:
            normals = normalise(normals)
        features = np.hstack((vertices, normals))
        print("--> {} {} vertices in total".format(vertices.shape[0], o_type))
    return torch.Tensor(features).transpose(0, 1) # N x 6

#Simple FC NN for optimization based nst (or use SIREN decoder?)
class SimpleFCNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = 3
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers + 2):
            if i == 0:
                layers.append(nn.Linear(self.latent_dim, self.hidden_dim))
            elif i == num_layers + 1:
                layers.append(nn.Linear(self.hidden_dim, self.out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def run_sdf():
    # experiment paths
    encoder_path = opt.pretrained_enc
    pretrain_path = opt.pretrained_sdf
    nst_output_path = opt.output_dir
    #nst_dir = os.path.join(nst_output_path, 'results')
    #summary_dir = os.path.join(nst_output_path, 'summary')

    # load data
    print("loading NST data ...")
    # the directory that holds the pre-processed (sampled) points
    O_c = get_samples(opt.content_shape, "content")
    O_s = get_samples(opt.style_shape, "style")

    #dataloader?
    # initialise training configs
    model = load_model(pretrain_path, "sdf").to(device)
    encoder = load_model(encoder_path, "encoder")
    batch_size = 16
    num_epochs = opt.num_epochs
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)  # natural gradients maybe ?
    content_loss = nn.MSELoss()
    style_loss = loss.StatsLoss()
    style_weight = 15

    visualise_per_k_iter = 50
    save_every = 200
    reconstruct_every = 100 # reconstrcut the obj after {} steps of gradient updates
    assert reconstruct_every < save_every # otherwise the model has not been changed

    # features, fixed dict
    E_Oc = encoder(O_c) # normalization done in voxelization.py
    E_Os = encoder(O_s)

    # initial reconstruction, point features and global features are extracted
    print("initial reconstruction")
    E_O_tilde = get_mesh_features(encoder, model, nst_output_path)

    # main training loop
    # dataloader must be used when integrating SIREN for feedforwards training
    print('Starting NST')
    start = time.time()
    model.train()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    for i in tqdm(range(num_epochs), desc="training nst"):
        if i % reconstruct_every == 0 and i > 0:
            E_O_tilde = get_mesh_features(encoder, model, nst_output_path)

        if i % save_every:
            torch.save(model.state_dict(),
                       os.path.join(nst_output_path, 'model_epoch%04d.pth' % i))

        # content_loss = loss(E_Oc["global_feature"], E_O_tilde["global_feature"])
        content_losses = torch.tensor([content_loss(e_oc, e_ot) for (e_oc, e_ot) in zip(E_Oc.values(), E_O_tilde.values())])
        style_losses = torch.tensor([style_loss(e_os, e_ot) for (e_os, e_ot) in zip(E_Os.values(), E_O_tilde.values())])
        content_loss = torch.sum(content_losses)
        style_loss = torch.sum(style_losses)
        total_loss = content_loss + style_weight * style_loss

        print('epoch: {}, loss: {}'.format(i, total_loss))

        total_loss.backward()
        optimizer.step()
    end = time.time()
    sec_elapsed = end-start
    print('NST finished, time elapsed: {}'.format(sec_elapsed))

    torch.save(model.state_dict(),
                       os.path.join(nst_output_path, 'model_latest.pth'))
    print('Saved latest model in directory: {}'.format(nst_output_path))



def run_optimization_based():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir

    # load data
    print("loading NST data ...")
    # the directory that holds the pre-processed (sampled) points
    O_c = get_samples(opt.content_shape, d_type="content")
    O_s = get_samples(opt.style_shape, d_type="style")
    encoder = load_model(encoder_path, "encoder")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # sample same number of points and normalise into [-1, 1]
    num_samples = opt.num_points
    print("using %d samples" % num_samples)
    ids_c = np.random.choice(O_c.shape[1], num_samples) # not necessarily need to normalise
    ids_s = np.random.choice(O_s.shape[1], num_samples) # normed: shapeNet data -> PVCNN input -> save output
    O_c_, O_s_ = O_c[:, ids_c], O_s[:, ids_s]

    O_c_ = torch.unsqueeze(O_c_, 0).to(device)
    O_s_ = torch.unsqueeze(O_s_, 0).to(device)

    E_Oc, _ = encoder(O_c_)
    E_Os, _ = encoder(O_s_) # [1, f, N]

    # initialise object
    random = False
    if random:
        pass
    else:
        obj = O_c_.clone() # use the same O_c_ or re-sample?
    obj = nn.Parameter(obj.to(device), requires_grad=True)

    # training
    num_epochs = opt.num_epochs
    learning_rate = 5e-3
    optimizer = torch.optim.Adam(params=[obj], lr=learning_rate)
    content_loss = nn.MSELoss().to(device)
    style_loss = loss.StatsLoss().to(device)
    style_weight = opt.style_weight
    save_every = opt.save_every
    writer = SummaryWriter()

    # save stylised object
    def save_to(params, suffix):
        pcd = o3d.geometry.PointCloud()
        pts = params.detach().cpu()
        pcd.points = o3d.utility.Vector3dVector((pts.numpy())[:, 0:3])
        save_path = os.path.join(nst_output_path, "stylization_it{}.ply".format(suffix))
        print(" --> saving to ", save_path)
        o3d.io.write_point_cloud(save_path, pcd)

    print('Starting NST')
    start = time.time()
    content_feats, style_feats = [], []
    for _, v in E_Oc.items():
        content_feats.append(v.detach())
    for _, v in E_Os.items():
        style_feats.append(v.detach())

    # batch_size = 32
    # iters = int(num_samples/batch_size)

    for n in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        if n % save_every == 0 and n > 0:
            save_to(torch.squeeze(obj).transpose(0, 1).detach(), n)

        E_O_tilde, _ = encoder(obj) # change this to batchload after
        stylisesed_feats = []
        content_losses, style_losses = [], []

        for _, v in E_O_tilde.items():
            stylisesed_feats.append(v)

        for i in range(len(stylisesed_feats)):
            cf = content_feats[i]
            sf = style_feats[i]
            styf = stylisesed_feats[i]
            content_losses.append(content_loss(cf, styf))
            style_losses.append(style_loss(sf, styf))

        cl = sum(content_losses)
        sl = sum(style_losses)
        total_loss = cl + style_weight * sl

        writer.add_scalar("content loss", cl, n)
        writer.add_scalar("weighted style loss", style_weight*sl, n)
        writer.add_scalar("total loss", total_loss, n)
        writer.add_scalar("obj sum", obj.sum(), n)

        #print('epoch: {}, loss: {}'.format(n, total_loss))
        total_loss.backward()
        optimizer.step()
    end = time.time()
    print('training time: ', end-start)

    # save model for reconstruction
    save_to(torch.squeeze(obj).transpose(0, 1).detach(), "_final")


def run_layers():
    # experiment paths
    encoder_path = opt.pretrained_enc
    nst_output_path = opt.output_dir

    # load data
    print("loading NST data ...")
    # the directory that holds the pre-processed (sampled) points
    O_c = get_samples(opt.content_shape, d_type="content")
    O_s = get_samples(opt.style_shape, d_type="style")
    encoder = load_model(encoder_path, device=device, option="encoder")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # sample same number of points and normalise into [-1, 1]
    num_samples = opt.num_points
    print("using %d samples" % num_samples)
    ids_c = np.random.choice(O_c.shape[1], num_samples)  # not necessarily need to normalise
    ids_s = np.random.choice(O_s.shape[1], num_samples)  # normed: shapeNet data -> PVCNN input -> save output
    O_c_, O_s_ = O_c[:, ids_c], O_s[:, ids_s]

    # append shape category
    category_s, category_c = torch.zeros(16), torch.zeros(16)
    category_c[cat2id(opt.content_category)] = 1
    category_s[cat2id(opt.style_category)] = 1
    category_c = category_c.repeat(num_samples, 1).t()
    category_s = category_c.repeat(num_samples, 1).t()
    O_c_ = torch.vstack((O_c_, category_c))
    O_s_ = torch.vstack((O_s_, category_s))

    O_c_ = torch.unsqueeze(O_c_, 0).to(device)
    O_s_ = torch.unsqueeze(O_s_, 0).to(device)
    E_Oc, scores_Oc = encoder(O_c_)
    E_Os, scores_Os = encoder(O_s_)  # [1, f, N]

    # initialise object
    random = False
    if random:
        pass
    else:
        obj_ = O_c_.clone()  # use the same O_c_ or re-sample?

    # layers config
    '''
    feature_0...4: intermediate features
    feature_5: global feature

    features are of size [B, S, num_points], S \in [64, 128, 512, 2048]
    '''
    layers = opt.layers
    # layers = [int(e) for e in layers]

    content_feats, style_feats = [], []

    for k, v in E_Oc.items():
        if int(k[8]) in layers:
            content_feats.append(v.detach())
    for k, v in E_Os.items():
        if int(k[8]) in layers:
            style_feats.append(v.detach())

    # reuse cloned obj
    if opt.randn_noise:
        obj = nn.Parameter(torch.randn_like(obj_).to(device), requires_grad=True)
    else:
        obj = nn.Parameter(obj_.to(device), requires_grad=True)

    # training configs
    exp_name = opt.layers_exp_suffix
    num_epochs = opt.num_epochs
    learning_rate = opt.learning_rate

    early_stopping, early_stopping_thresh = opt.early_stopping, opt.early_stopping_thresh
    lr_decrease, rate = opt.lr_decrease, opt.lr_dec_rate
    optimizer = torch.optim.Adam(params=[obj], lr=learning_rate)

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

    for n in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        if n % save_every == 0 and n > 0:
            save_to(torch.squeeze(obj).transpose(0, 1).detach(), n, path=nst_output_path)

        E_O_tilde, _ = encoder(obj)
        stylised_feats = []
        content_losses, style_losses = [], []

        for k, v in E_O_tilde.items():
            if int(k[8]) in layers:
                stylised_feats.append(v)

        for i in range(len(stylised_feats)):
            cf = content_feats[i]
            sf = style_feats[i]
            styf = stylised_feats[i]
            # ch_dist_o_c = chamfer_loss(torch.transpose(O_c_, 1, 2), torch.transpose(obj, 1, 2))
            # ch_dist_o_s = chamfer_loss(torch.transpose(O_s_, 1, 2), torch.transpose(obj, 1, 2))
            content_losses.append(mse_loss(cf, styf))  # + ch_dist_o_c + ch_dist_o_s)
            style_losses.append(style_loss(sf, styf, dim=0 if layers[i] == 5 else 1))

        cl = sum(content_losses)
        sl = sum(style_losses)
        total_loss = content_weight * cl + style_weight * sl
        if n > 0 and early_stopping:
            if abs(total_loss - prev_total_loss) < early_stopping_thresh:
                print("early stopping threshold reached, stopping training")
                break

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
            # writer.add_scalar("{}/parts loss".format(exp_name), parts_loss, n)
            # writer.add_scalars("{}/chamfer losses".format(exp_name),
            #                    {'chamfer distance o <-> o_c': ch_dist_o_c,
            #                     'chamfer distance o <-> o_s': ch_dist_o_s},
            #                    n)
            writer.add_scalar("{}/optimal iter for total loss".format(exp_name), opt_iter, n)

        if verbose:
            print("iter {}, total loss: {}".format(n, total_loss))
        total_loss.backward()
        optimizer.step()
    end = time.time()
    print('training time: ', end - start)

    # save model for reconstruction
    save_to(torch.squeeze(obj).transpose(0, 1).detach(), "_final_{}".format(exp_name), path=nst_output_path)
