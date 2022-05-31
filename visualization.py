import open3d as o3d
import numpy as np
import matplotlib.cm as cm

import os

def interpolate(pts1, pts2, alpha):
    """
    reads two open3d point clouds
    """
    assert alpha >= 0 and alpha <= 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts1 * alpha + pts2 * (1-alpha))
    return pcd


def visualise_interpolation(obj1, obj2, view='consec', n=500):
    """
    visualises n interpolations (n controls the speed of visualization)

    obj1, obj2 are open3d point cloud objects

    valid arguments for view:
        -'single': shows single interpolation objects (need to manually clos)
        -'consec': shows interpolation as animation
    """
    pts1 = np.asarray(obj1.points)
    pts2 = np.asarray(obj2.points)

    num_samples = 30000
    pts1_ = pts1[np.random.choice(pts1.shape[0], num_samples), :]
    pts2_ = pts2[np.random.choice(pts2.shape[0], num_samples), :]

    if view == 'consec':
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(obj1) # init
        inter_obj_last = obj1
        opt = vis.get_render_option()
        opt.point_size = 3.5

    step = 1./n

    for i in range(n):
        inter_obj = interpolate(pts1_, pts2_, alpha=i*step)
        if view == 'single':
            o3d.visualization.draw_geometries([inter_obj])
        elif view == 'consec': #TODO: add interactive
            vis.remove_geometry(inter_obj_last)
            vis.add_geometry(inter_obj)
            vis.update_geometry(inter_obj)
            vis.update_renderer()
            vis.poll_events()
            inter_obj_last = inter_obj
    if view == 'consec':
        vis.destroy_window()


def visualise_rotation(pcd, speed=5.):
    """
    visualizes the rotation of an obj

    {pcd} is of type open3D PointCloud
    {speed} controls the angle speed
    """
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(speed, 0.0)
        return False
    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

def visualise_color(obj, color_axis=0, rgb=None):
    """
    visualise obj with rainbow color along a given axis, or use {rgb}
    """
    pts = np.asarray(obj.points)

    if color_axis >= 0:
        if color_axis == 3:
            axis_vis = np.arange(0, pts.shape[0], dtype=np.float32)
        else:
            axis_vis = pts[:, color_axis]
        min_ = np.min(axis_vis)
        max_ = np.max(axis_vis)

        colors = cm.gist_rainbow((axis_vis - min_) / (max_ - min_))[:, 0:3]
        obj.colors = o3d.utility.Vector3dVector(colors)
    if rgb is not None:
        obj.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([obj])


if __name__ == '__main__':
    o1 = o3d.io.read_point_cloud("preprocessed_meshes/paired/lamp_parts/lamp_parts_content_sbd.xyz")
    o2 = o3d.io.read_point_cloud("logs/parts/2000/lamp_parts_w5/0/stylization_it_final_all.ply")
    #visualise_color(o2, color_axis=0)
    #visualise_interpolation(o2, o1, view='consec', n=1500)

    # fs = []
    # for f in os.listdir(path):
    #     if 'ply' in f:
    #         fs.append(f)
    # fs = sorted(fs)
    #
    # for i in range(len(fs)):
    #     fs[i] = o3d.io.read_point_cloud(os.path.join(path, fs[i]))
    #
    # num_samples = 30000
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.point_size = 3.5
    # n = 500
    # step = 1./n
    #
    # for j in range(len(fs)-1):
    #     pts0 = np.asarray(fs[j].points)
    #     pts1 = np.asarray(fs[j+1].points)
    #     pts0_ = pts0[np.random.choice(pts0.shape[j], num_samples), :]
    #     pts1_ = pts0[np.random.choice(pts1.shape[j+1], num_samples), :]
    #     vis.add_geometry(fs[j])  # init
    #     inter_obj_last = fs[j]
    #     for i in range(n):
    #         inter_obj = interpolate(pts0_, pts1_, alpha=i*step)
    #         vis.remove_geometry(inter_obj_last)
    #         vis.add_geometry(inter_obj)
    #         vis.update_geometry(inter_obj)
    #         vis.update_renderer()
    #         vis.poll_events()
    #         inter_obj_last = inter_obj

