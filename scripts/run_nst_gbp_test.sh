# 4 experiments
SCRIPT_PATH="run_nst.py"

LAMP="lamp"
TABLE="table"

CONTENT_OBJ="preprocessed_meshes/paired/lamp_parts_1/lamp_parts_1_content_sbd.xyz"
STYLE_OBJ="preprocessed_meshes/paired/lamp_parts_1/lamp_parts_1_style_sbd.xyz"

args=(
      --mode ''
      --num_epochs 1
      --save_every 200
      --content_shape $CONTENT_OBJ
      --style_shape $STYLE_OBJ
      --content_category $LAMP
      --style_category $LAMP
      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
      --output_dir "logs/lamp_parts/1"
      --content_weight 1
      --style_weight 15
      --num_points 32000
      --layers_exp_suffix "all"
  )

python $SCRIPT_PATH "${args[@]}"



#CONTENT_OBJ="preprocessed_meshes/paired/lamp_parts_2/lamp_parts_2_content_sbd.xyz"
#
#STYLE_OBJ="preprocessed_meshes/paired/lamp_parts_2/lamp_parts_2_style_sbd.xyz"
#
#args=(
#      --mode 'gbp'
#      --num_epochs 2000
#      --save_every 200
#      --content_shape $CONTENT_OBJ
#      --style_shape $STYLE_OBJ
#      --content_category $LAMP
#      --style_category $LAMP
#      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#      --output_dir "logs/lamp_parts/2"
#      --content_weight 1
#      --style_weight 15
#      --num_points 32000
#      --layers_exp_suffix "all"
#  )
#
#bsub -n 4 -W 2:00 -R  "rusage[mem=2048,ngpus_excl_p=4]" python $SCRIPT_PATH "${args[@]}"
#
#
#CONTENT_OBJ="preprocessed_meshes/paired/table_parts_1/table_parts_1_content_sbd.xyz"
#
#STYLE_OBJ="preprocessed_meshes/paired/table_parts_1/table_parts_1_style_sbd.xyz"
#
#args=(
#      --mode 'gbp'
#      --num_epochs 2000
#      --save_every 200
#      --content_shape $CONTENT_OBJ
#      --style_shape $STYLE_OBJ
#      --content_category $TABLE
#      --style_category $TABLE
#      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#      --output_dir "logs/table_parts/1"
#      --content_weight 1
#      --style_weight 15
#      --num_points 32000
#      --layers_exp_suffix "all"
#  )
#
#bsub -n 4 -W 2:00 -R  "rusage[mem=2048,ngpus_excl_p=4]" python $SCRIPT_PATH "${args[@]}"
#
#
#
#CONTENT_OBJ="preprocessed_meshes/paired/table_parts_2/table_parts_2_content_sbd.xyz"
#
#STYLE_OBJ="preprocessed_meshes/paired/table_parts_2/table_parts_2_style_sbd.xyz"
#
#args=(
#      --mode 'gbp'
#      --num_epochs 2000
#      --save_every 200
#      --content_shape $CONTENT_OBJ
#      --style_shape $STYLE_OBJ
#      --content_category $TABLE
#      --style_category $TABLE
#      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#      --output_dir "logs/table_parts/2"
#      --content_weight 1
#      --style_weight 15
#      --num_points 32000
#      --layers_exp_suffix "all"
#  )
#
#bsub -n 4 -W 2:00 -R  "rusage[mem=2048,ngpus_excl_p=4]" python $SCRIPT_PATH "${args[@]}"
#
#
#
#
