"""
Default parameters for train_nst_clf argparse
"""

# io configs
save_every = 500
output_dir = 'logs/nst_clf_test'
exp_name = 'test'
with_color = False

# data configs
content_class = 'lamp'
style_class = 'lamp'
c_path = 'data/PartAnnotation/03636649/points/1a6a520652aa2244146fa8a09fad6c38.pts'
s_path = 'data/PartAnnotation/03636649/points/1a9c1cbf1ca9ca24274623f5a5d0bcdc.pts'
randn_noise = True
normalize = True
num_points = 512

# model configs
pretrained_enc = "pretrained_models/clf_ep_24_maj.pth"

# optimization configs (for opt. based)
num_epochs = 1000
content_weight = 0
style_weight = 1
attribution_weight = 5
learning_rate = 1e-3

early_stopping = False
early_stopping_thresh = 1e-6

# not implemented yet
lr_decrease = False
lr_dec_rate = .5

# optimizer configs

# method
mode_list = ['gbp',  'int_grad', 'grad_cam']
mode = 'gbp'

# PointNet configs
content_layers = [0]
style_layers = [0]

# summary configs
verbose = True
summarize = False



