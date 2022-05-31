# testing ground
SCRIPT_PATH="train_nst.py"

CAT="lamp"

CONTENT_OBJ="preprocessed_meshes/paired/lamp_parts_5/lamp_parts_5_content_sbd.xyz"
STYLE_OBJ="preprocessed_meshes/paired/lamp_parts_5/lamp_parts_5_style_sbd.xyz"

args=(
      --mode ''
      --num_epochs 2
      --save_every 50
      --content_shape $CONTENT_OBJ
      --style_shape $STYLE_OBJ
      --content_category $CAT
      --style_category $CAT
      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
      --output_dir "logs/lamp_color_3_debug" #base to lampshade
      --content_weight 1
      --style_weight 10
      --num_points 2048
      --layers_exp_suffix "all"
      --with_color 1
      --normalize 1
  )

python $SCRIPT_PATH "${args[@]}"

