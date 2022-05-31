import numpy as np
import open3d as o3d
import torch
import seaborn as sns

import os
import glob
import argparse
from collections import OrderedDict

import shapenet_constants
from modules.pointnet import PointNet
from modules.pointnet_classifier import PointNetClf

__all__ = ["ExperimentParser", "cat2id", "get_part_info", "get_part_id", "get_id2part", "get_shapename2id",
           "normalise", "norm_unit", "get_samples", "load_model", "save_to", "sample_uniform_from", "standardize",
           "color_attribution", "color_samples"]






'''
EXPERIMENT SETUP
'''
class ExperimentParser():
    # TODO: condition violation checking/subparsers
    """
    A wrapper for ArgumentParser
    """
    def __init__(self):
        p = argparse.ArgumentParser()

        # io configs
        p.add_argument('--output_dir', type=str, default='logs/new_experiment',
                       help='The output directory, if not specified, the results will be stored \ '
                            'to logs/experiment. Results include: \n '
                            'i) the stylization result in .ply format \n  '
                            'ii) the model parameters of the stylization result \n'
                            'iii) the torch tensorboard summaries')
        p.add_argument('--layers_exp_suffix', type=str, default='',
                       help='suffix for final stylization (layer experiments)')
        p.add_argument('--with_color', type=int, default=0,
                       help='whether to use color, default false')

        # data configs
        p.add_argument('--content_shape', type=str, required=True,
                       help='directory to the content shape ')
        p.add_argument('--content_category', type=str, required=True,
                       help='category of the content shape ')
        p.add_argument('--style_shape', type=str, required=True,
                       help='directory to the style shape ')
        p.add_argument('--style_category', type=str, required=True,
                       help='category of the content shape ')
        p.add_argument('--randn_noise', type=bool, default=False,
                       help='whether to initialise the content shape u.a.r')
        p.add_argument('--normalize', type=int, default=1,
                       help='normalize sampled points into [-1, 1]')

        # model configs
        p.add_argument('--pretrained_enc', type=str, required=True,
                       help='directory to the pretrained feature extractor (.pth)')

        # training configs
        p.add_argument('--num_iters', type=int, default=1000,
                       help='Number of training epochs')
        p.add_argument('--save_every', type=int, default=100,
                       help='The number of iterations to save the stylization result (used by \
                       run_optimization_based())')
        p.add_argument('--content_weight', type=float, default=1,
                       help='Weight of the content loss in the optimization based NST')
        p.add_argument('--style_weight', type=float, default=15,
                       help='Weight of the style loss in the optimization based NST')
        p.add_argument('--num_points', type=int, default=32000,
                       help='number of points to sample from the object')

        # optimizer configs
        p.add_argument('--step_size', type=float, default=1e-2,
                       help='step size for Adam')

        # method
        mode_list = ['score_plain', 'gbp', 'int_grad', 'ada_attn', 'template']
        p.add_argument('--mode', type=str, default='gbp',
                       help='the part-agnostic masking method to apply. one of \ '
                            '{}'.format(mode_list))

        # nst model

        # encoder configs
        p.add_argument('--layers', nargs='+', default=[0, 1, 2, 3, 4, 5],
                       help='a list of PointNet feature layers to use; \ '
                            '0-4: local features, 5: global features')

        # summary configs
        p.add_argument('--verbose', action=argparse.BooleanOptionalAction)
        p.add_argument('--summarize', action=argparse.BooleanOptionalAction)

        self.opt = p





'''
DATASET PROPERTIES
'''

part2id = {
    0:"body",
    1:"wing",
    2:"tail",
    3:"engine",
    # 4:
    # 5:
    # 6:
    # 7:
    8:"roof",
    9:"hood",
    10:"wheel",
    11:"body",
    12:"back",
    13:"seat",
    14:"legs",
    15:"arms",
    24:"base",
    25:"lampshade",
    26:"canopy",
    27:"lamppost",
    36:"handle",
    37:"body",
    47:"tabletop",
    48:"legs"
}

def cat2id(cat):
    """
    converts shape categories into the corresponding id used for ShapeNet part training
    """
    assert isinstance(cat, str)
    return shapenet_constants.ShapeId[cat.upper()].value


def get_part_info(obj):
    """
    get number of parts of an object and the names of the parts

    :param obj: is a string representing the shape category
    :return: a dict of: number of parts, a list of the part names sorted alphabetically
    """
    info = {}
    shape_dir = os.path.join('data/PartAnnotation', get_shapename2id()[obj.upper()], 'points_label')
    if os.path.exists(os.path.join(shape_dir, '.DS_Store')):
        os.system("rm {}".format(os.path.join(shape_dir, '.DS_Store')))
    info['num_parts'] = len(os.listdir(shape_dir))
    info['part_names'] = sorted(os.listdir(shape_dir))
    print(info['part_names'])
    return info


def get_part_id(obj):
    """
    get part ids of an object

    :param obj: is a string representing the shape category
    :return: a torch.Tensor of the part ids
    """
    shapeid = get_shapename2id()[obj.upper()]
    return torch.tensor(_get_partname2partid()[shapeid])


def get_id2part(obj):
    part_names = get_part_info(obj)['part_names']
    part_ids = get_part_id(obj)
    return {id: part for id, part in zip(part_ids, part_names)}


def _get_partname2partid():
    """
    converts part names into the corresponding part id
    TODO: reindex this
    """

    data_dir = 'data/PartAnnotation'
    part2id = {}
    shape_anno = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    id = 0
    for shapeid in sorted(shape_anno):
        part_ids = []
        for f in sorted(os.listdir(os.path.join('data/PartAnnotation', shapeid, 'points_label'))):
            if f == '.DS_Store':
                continue
            part_ids.append(id)
            id += 1
        part2id[shapeid] = part_ids
    return part2id


def get_shapename2id():
    """
    converts part categories into the corresponding id
    """
    name2id = {}
    with open(os.path.join('data/PartAnnotation', 'synsetoffset2category.txt'), 'r') as f:
        for shape_id, line in enumerate(f):
            shape_name, shape_dir = line.strip().split()
            name2id[shape_name.upper()] = shape_dir
    return name2id





'''
DATA PROCESSING
'''
def sample_uniform_from(obj, num_samples, replace=False):
    """
    obj must be of the shape [3, num_points]
    """
    ids = np.random.choice(obj.shape[1], num_samples, replace=replace)
    sampled_obj = obj[:, ids]
    return sampled_obj


def normalise(coords, ax=1):
    """
        normalises 3D coordinates into [-1, 1]

        shape [_, N] -> ax = 1
        1d vector -> ax = 0
    """
    #print("pts shape: ", coords.shape)
    if isinstance(coords, np.ndarray):
        coords -= np.mean(coords, axis=ax, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
    elif isinstance(coords, torch.Tensor):
        coords -= torch.mean(coords, axis=ax, keepdim=True)
        coord_max = torch.amax(coords)
        coord_min = torch.amin(coords)
    else:
        raise RuntimeError("unsupported type: ", type(coords))
    coords = (coords - coord_min) / (coord_max - coord_min + 1e-12)
    coords -= 0.5
    coords *= 2.
    return coords

def norm_unit(coords):
    """
    normalizes coordinates into [0, 1]
    """
    coord_max,  coord_min = torch.amax(coords), torch.amin(coords)
    coords = coords - coord_min
    coords = coords / (coord_max + 1e-12)
    return coords

def standardize(points):
    """
    standardizes input points (e.g. coordinates)
    """
    points_std = torch.std(points)
    points_mean = torch.mean(points)
    points = points - points_mean
    points = points / (points_std + 1e-12)
    return points


#TODO: 1.can be truncated
#TODO: 2.can put all normalization code here
def get_samples(path, d_type=None, with_normals=False, o_type="pcd", with_normalization=False, to_tensor=True,
                **kwargs):
    """
    # read 3D points to mesh for feature extraction
    :param path: path to the mesh
    :param d_type: input data type, if 'txt' open3d will not be used
    :param with_normals:
    :param o_type: output type
    :param to_tensor: if True, converts the read samples to torch.Tensor,
                       otherwise returns a numpy array
    kwargs:
        -'subdivide': if set to True, subdivides the loaded mesh until it possesses at least 'num_vert' vertices
    :return: sampled points from the object of type np.array or torch.Tensor
    """
    if d_type == "txt":
        features = np.loadtxt(path).astype(np.float32)
    else:
        if o_type == "pcd":
            print("reading {} points ...".format(o_type))
            pcd = o3d.io.read_point_cloud(path)
            vertices = np.asarray(pcd.points)
            if with_normals:
                normals = np.asarray(pcd.normals)
                features = np.hstack((vertices, normals))
            else:
                 features = vertices
            print("--> {} {} points in total".format(vertices.shape[0], o_type))
        elif o_type == "trimesh":
            print("reading {} vertices ...".format(o_type))
            mesh = o3d.io.read_triangle_mesh(path)
            if np.asarray(mesh.vertices).shape[0] == 0:
                raise IOError("0 vertices found on path {}. Please check your directory or object file".format(path))
            if kwargs['subdivide']:
                while np.asarray(mesh.vertices).shape[0] < kwargs['num_vert']:
                    print("Target has less than {} vertices ({}). Performing subdivision".format(\
                          kwargs['num_vert'], np.asarray(mesh.vertices).shape[0]))
                    mesh = mesh.subdivide_loop(1)
            vertices = np.asarray(mesh.vertices)

            if with_normals:
                mesh.compute_vertex_normals()
                normals = np.asarray(mesh.triangle_normals)
                features = np.hstack((vertices, normals))
            else:
                features = vertices
            print("--> {} {} vertices in total".format(vertices.shape[0], o_type))
        else:
            raise IOError("unknown data type: ", o_type)

    # to 3 X N
    if to_tensor:
        samples = torch.Tensor(features).transpose(0, 1)
    else:
        samples = features.transpose(1, 0)

    if with_normalization:
        samples = normalise(samples, ax=1)
    return samples


def load_model(path, device, option='segmentation', **kwargs):
    """
    Load a model to device
    :param path: path to the pretrained model's checkpoint
    :param device: the device to which the model should be loaded onto
    :param option: type of the model, one of ['encoder', 'decoder']

    **kwargs specifies:
        -use_clf: whether to use the part segmentation scores
        -global_pooling: that pooling scheme to use for the last layer ('max', 'avg')
        -new_ckpt: creates a new StateDict that has the "model." prefix removed
    """
    print("loading {} model".format(option))
    if option == 'segmentation':
        model = PointNet(
            num_classes=shapenet_constants.NUM_PART_CATEGORIES,
            num_shapes=shapenet_constants.NUM_SHAPE_CATEGORIES,
            extra_feature_channels=0, # pointnet takes Nx3 shape input by default
            global_pooling=kwargs['global_pooling'],
            use_clf=kwargs['use_clf'],
            retain_feature_grad=kwargs['retain_feature_grad']
        )
    elif option == 'classification':
        model = PointNetClf(
            k=kwargs['k'],
            with_transformer=True,
            global_pooling=kwargs['global_pooling'],
            use_clf=kwargs['use_clf'],
            retain_feature_grad=kwargs['retain_feature_grad']
        )
    else:
        raise RuntimeError("unidentified option: ", option)

    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device(device))
    else:
        checkpoint = torch.load(path)

    # do this if the model checkpoint has not been saved in a "regular" way
    if kwargs['new_ckpt']:
        new_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k[:7] == 'module.':
                name = k[7:]
            else:
                name = k
            new_dict[name] = v
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    print("load state dict done")
    return model


def save_to(params, suffix, path, format='ply', with_color=True):
    """
    save stylised object to path with a suffix in a specified format
    :param format: one of ['ply', 'xyz', 'xyzn'(not implemented yet), 'xyzrgb']

    if with_color=True, the input object should contain the points' colors
    """
    pcd = o3d.geometry.PointCloud()
    pts = params.cpu()

    if with_color:
        pcd.points = o3d.utility.Vector3dVector((pts[:, :3].numpy()))
        pcd.colors = o3d.utility.Vector3dVector((pts[:, 3:].numpy()))
    else:
        pcd.points = o3d.utility.Vector3dVector((pts.numpy()))

    if not os.path.exists(path):
        os.system("mkdir {}".format(path))
    save_path = os.path.join(path, "stylization_it{}.{}".format(suffix, format))
    print(" --> saving to ", save_path)
    o3d.io.write_point_cloud(save_path, pcd)





'''
VISUALIZATION
'''
#color lists
custom_color_list1 = [
    (1., 0., 0.),  # red
    (0., 1., 0.),  # green
    (0., 0., 1.),  # blue
    (1., 1., 0.),  # yellow
    (1., 165./255, 0.),  # orange
    (1., 105./255, 180./255),  # hotpink
    (0., 0., 0.),  # black
    (.5, 0, .8),  # purple
    (.9, .8, 1),  # lavender
    (1, .9, .3),  # dark yellow
    (.4, 1, 1)  # cyan
]
custom_color_list2 = sns.color_palette("husl", shapenet_constants.NUM_PART_CATEGORIES)#shapenet_constants.MAX_PARTS)
color_lists = [custom_color_list1, custom_color_list2]


def color_attribution(pts, mode, attribution, **kwargs):
    """
    :param mode: attribution mode, one of ['gradients', 'displacement']
    :param attribution: the attribution features for coloring. Of size [feat_size, num_points] / [3, num_points]

    For gradients:
        kwargs['color'] = 'negpos' gives:
        red -> positive contribution
        blue -> negative contribution

        kwargs['color'] = 'magnitude' gives:
        red -> more contribution

    For displacement:
        red -> distance

    :returns A colored point cloud
    """

    # make colors
    if mode == 'gradients':
        gradients = attribution
        if len(gradients.shape) == 2:
            gradients_ = gradients.detach().transpose(0, 1).cpu()


        if kwargs['color'] == 'negpos':
            if len(gradients.shape) == 2:
                grad_aggr = torch.max(gradients_, dim=1).values #mean(gradients_, dim=1)#
                grad_aggr = normalise(grad_aggr, ax=0)
                colors = torch.zeros(gradients_.shape[0], 3)
            else: # for gradCAM
                grad_aggr = gradients.detach().cpu()
                colors = torch.zeros(gradients.shape[0], 3)
            for i in range(colors.shape[0]):
                colors[i, :] = torch.tensor([grad_aggr[i], 0., 0.]) if grad_aggr[i] > 0 else \
                    torch.tensor([0., 0., -grad_aggr[i]])

        elif kwargs['color'] == 'magnitude':
            grad_mag = torch.norm(gradients_, dim=1)
            grad_mag = norm_unit(grad_mag)
            colors = torch.zeros(gradients_.shape[0], 3)
            for i in range(colors.shape[0]):
                colors[i, :] = torch.tensor([grad_mag[i], 0., 0.])

        else:
            raise IOError("color mode {} does not exist for {}".format(kwargs['color'], mode))

    elif mode == 'displacement':
        assert len(attribution.shape) > 1 #TODO
        moved_pts = attribution
        displacement = torch.norm(moved_pts-pts, dim=0)
        min_val, max_val =  torch.amin(displacement), torch.amax(displacement)
        displacement -= min_val
        displacement /= (max_val + 1e-12)
        colors = torch.zeros(displacement.shape[0], 3)
        colors[:, 0] = displacement

    else:
        raise TypeError("Attribution mode not found: ", mode)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((pts.transpose(0, 1).detach()[:, :3].cpu().numpy()))
    pcd.colors = o3d.utility.Vector3dVector((colors.detach().cpu().numpy()))

    return pcd


def color_samples(obj, encoder, save_path=None, visualize=False):
    """
    saves the obj along with the colors corresponding to part scores
    """
    _, scores = encoder(obj)  # 1, 50, N
    part_classes = torch.max(torch.squeeze(scores), dim=0).indices
    num_samples = scores.shape[2]
    colors = torch.zeros(num_samples, 3)

    col_list = color_lists[0]

    print(col_list)
    for i in range(colors.shape[0]):
        id = part_classes[i] % len(col_list)
        print("{}->{}".format(part_classes[i], id))
        colors[i, :] = torch.tensor(col_list[id])

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(((torch.squeeze(obj, 0)).transpose(0, 1).detach()[:, :3].cpu().numpy()))
        pcd.colors = o3d.utility.Vector3dVector((colors.cpu().numpy()))
        o3d.visualization.draw_geometries([pcd])

    obj = torch.hstack((torch.squeeze(obj[:, :3, :]).transpose(0, 1).detach(), colors))
    if save_path is not None:
        save_to(obj, 'test', save_path, format='ply', with_color=True)


def _save_part_seg_color(load_path, save_path, cat_name):
    """
    Loads a 3D object of category {cat_name} from a path and s
    aves the colored part segmentation result
    """
    obj = o3d.io.read_point_cloud(load_path)
    pts = torch.tensor((np.asarray(obj.points)))
    num_pts = pts.shape[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = load_model(path="pretrained_models/copy_shapenet.pointnet.pth.tar", device=device, option="encoder",
                         global_pooling='max', use_clf=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(get_id2part(cat_name))
    category = torch.zeros(shapenet_constants.NUM_SHAPE_CATEGORIES)
    category[cat2id(cat_name)] = 1
    category = category.repeat(num_pts, 1).t()

    pts = normalise(pts, ax=0)

    inp = torch.vstack((pts.t(), category))
    inp = inp.float()
    inp = torch.unsqueeze(inp, 0).to(device)
    color_samples(inp, encoder, save_path=save_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--obj_cat', type=str, required=True,
                   help='category of the object')
    parser = argparser.parse_args()
    obj_cat = parser.obj_cat

    print(get_part_id(obj_cat))
    print(part2id)