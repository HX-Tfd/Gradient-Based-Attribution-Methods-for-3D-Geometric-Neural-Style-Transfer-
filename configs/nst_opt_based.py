"""
Default parameters for train_nst argparse
Overwrite them in run_nst.py
"""

# io configs
output_dir = 'logs/test'
layers_exp_suffix = 'test'
with_color = False

# data configs
style_shape = 'data/PartAnnotation/04379243/points/1ac080a115a94477c9fc9da372dd139a.pts'
style_category = 'table'
content_shape = 'data/PartAnnotation/03001627/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts'
content_category = 'chair'
randn_noise = False
normalize = True

# model configs
pretrained_enc = 'pretrained_models/shapenet.pointnet.pth.tar' # path to pretrained feature extractor

# optimization configs (for opt. based)
num_iters = 5
save_every = 200
content_weight = 1
style_weight = 15
num_points = 1000
step_size = 1e-3
score_weight = 10  # for weighting score loss

# only used in GBP-based
gbp_mode = 'id' # one of [id, max, max_n]
part_id_c = None
part_id_s = None

# only used in GradCAM-based
style_only = False
content_only = False
full_loss = True
attr_weight = 10

# task-specific configs
part_specific = True        # you only need to set this when running run_nst_grad_cam()
mask_class_list = [25]      # these parts of the content object will be optimized
content_part_class = 25     # part id of the content object to be optimized
style_part_class = 47       # target part id of the style object

# optimizer configs

# method
mode_list = ['gbp']
mode = ''

# PointNet configs
content_layers = [0, 1, 2, 3, 4]
style_layers = [3, 4]

# summary configs
verbose = True      # prints loss per iteration
summarize = True    # creates tensorboard summary in output_dir



