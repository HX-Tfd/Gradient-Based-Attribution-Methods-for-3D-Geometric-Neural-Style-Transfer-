'''
Samples points from 3D objects
optionally performs mesh surface subdivision
'''


import torch
import open3d as o3d
import numpy as np
import os




'''
table0:
1a3cf7fca32910c4107b7172b5e0318e
1ab0b879f52e7cbe958d575443776c9b

table1:
1aba52edddfad70d1bf0233e4c77d163
1ab95754a8af2257ad75d368738e0b47

lamp:
1a6a520652aa2244146fa8a09fad6c38
1a9c1cbf1ca9ca24274623f5a5d0bcdc

bag:
45f56ad8c0f4059323166544c0deb60f
83ab18386e87bf4efef598dabc93c115

chair to table:
chair 1a6f615e8b1b5ae4dbbc9440457e303e
table 1a15e651e26622b0e5c7ea227b17d897

table2:
1a8a796da899cb2c4d672fe014b9000e
-> table 1a2914169a3a1536a71646339441ab0c
-> table 1ac080a115a94477c9fc9da372dd139a
-> table 1b739cf713702b2146f41dc2aaef556b
-> table 1bd138c3e54a75d32f38c0d2792fb5e
-> 1bd49921fac3e815c3bd24f986301745

comparison:
-> content [table] 1a8a796da899cb2c4d672fe014b9000e

-> style [table] 1adc25c3c29e98c454683b99ac4500e8
-> style [sofa] 1acdc3f794d927fd63fba60e6c90121a
-> style [lamp] 1a44dd6ee873d443da13974b3533fb59
-> style [guitar] 1edaab172fcc23ab5238dc5d98b43ffd


'''



def extract_mesh_comparison():
    '''
    extract meshes for cross-category experiments
    '''

    content_id = '1a8a796da899cb2c4d672fe014b9000e'
    content_category = 'table'

    style_id = ['1adc25c3c29e98c454683b99ac4500e8', '1acdc3f794d927fd63fba60e6c90121a',
             '1a6a520652aa2244146fa8a09fad6c38', '1edaab172fcc23ab5238dc5d98b43ffd']
    style_category = ['table', 'sofa', 'lamp', 'guitar']

    data_path = os.path.join(working_dir, "ShapeNetCore.v2")
    data_suffix = "models/model_normalized.obj"

    content_mesh_path = os.path.join(data_path, content_category, content_id, data_suffix)
    style_mesh_path = [os.path.join(data_path,
                                    style_category[i],
                                    style_id[i],
                                    data_suffix)
                       for i in range(len(style_id))]

    extract_mesh(content_mesh_path, 'content')
    for i in range(len(style_id)):
        extract_mesh(style_mesh_path[i], style_category[i])


def extract_mesh_single():
    '''
    extract single mesh
    '''
    exp_name = 'lamp_parts_3'
    save_path = os.path.join(working_dir, "preprocessed_meshes/paired", exp_name)
    category = 'lamp'
    content_id = '3a0719c32c45c16f96791035e86a30f4'
    style_id = '3a5a0f4c78e17b284f0c4075db76b7c'

    data_path = os.path.join(working_dir, "data/ShapeNetCore.v2")
    data_suffix = "models/model_normalized.obj"

    cpath = os.path.join(data_path, category, content_id, data_suffix)
    spath = os.path.join(data_path, category, style_id, data_suffix)
    extract_mesh(cpath, 'content', sbd_iters=4, save_path=save_path)
    extract_mesh(spath, 'style', sbd_iters=2, save_path= save_path)


def extract_mesh(data_path, attr, format='xyz', sbd_iters=None, save_path='preprocessed_meshes'):
    '''
    extract mesh as .xyz/.xyzn from {data_path} and saves to {save_path}
    optionally performs mesh subdivison
    '''

    # sanity checks
    if format not in ['xyz', 'xyzn']:
        raise NotImplementedError("no preprocessing implemented for output format {}".format(format))


    print("reading ", data_path)
    mesh = o3d.io.read_triangle_mesh(data_path)
    subdiv_sfx = ""
    if sbd_iters is None:
        subdiv_iters = subdivision_iterations
    else:
        subdiv_iters = sbd_iters

    if subdiv_iters > 1:
        print("subdivision specified for {} iterations".format(subdiv_iters))
        print("subdividing mesh ...")
        mesh = mesh.subdivide_loop(subdiv_iters)
        print("subdivision finished")
        subdiv_sfx = "sbd"
    else:
        print("runnning without mesh subdivision")
    print("total vertices: {}".format((((np.array(mesh.vertices)).shape)[0])))

    # export to .xyz/.xyzn
    print("convert into point cloud: ")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if format == 'xyzn':
        mesh.compute_vertex_normals()
        pcd.normals = mesh.vertex_normals # enable when necessary
    o3d.visualization.draw_geometries([pcd])
    print("done")

    # save obj
    # place holder
    obj_save_path = os.path.join(save_path, "{}_{}_{}{}.{}".format(exp_name, attr,
                                                                   subdiv_sfx, subdiv_iters, format))
    if not os.path.exists(obj_save_path):
        os.system("mkdir {}".format(obj_save_path))
    print("saving to {}".format(obj_save_path))
    o3d.io.write_point_cloud(obj_save_path, pcd)
    print("done")


if __name__ == 'main':
    from argparse import ArgumentParser

    p = ArgumentParser()

    p.add_argument('--data_path', type=str, required=True,
                   help="path to the object")
    p.add_argument('--save_path', type=str, default='preprocessed_meshes',
                   help="where to save the object")
    p.add_argument('--subdiv_iters', type=int, default=1,
                   help='Number of subdivision performed on the mesh')
    p.add_argument('--exp_name', type=str, default="experiment_0",
                   help='Name of the experiment, will be used to save the meshes')
    p.add_argument('--attr', type=str, default="",
                   help='optional attribute in the file name')
    p.add_argument('--working_dir', type=str, default=".",
                   help="the working directory")
    p.add_argument('--format', type=str, default="xyz",
                   help="the output format, one of [xyz, xyzn]")
    p.add_argument('--visualize', action='store_true',
                   help="visualize the object")
    opt = p.parse_args()

    working_dir = opt.working_dir

    subdivision_iterations = opt.subdiv_iters  # Number of subdivision performed on the mesh
    exp_name = opt.exp_name  # Name of the experiment, will be used to save the meshes
    working_dir = opt.working_dir  # the working directory
    data_path = opt.data_path
    attribute = opt.attr
    visualize = opt.visualize
    output_format = opt.format
    save_path = opt.save_path

    extract_mesh(data_path=data_path, attr=attribute, format=output_format, sbd_iters=subdivision_iterations,
                 save_path=save_path)


