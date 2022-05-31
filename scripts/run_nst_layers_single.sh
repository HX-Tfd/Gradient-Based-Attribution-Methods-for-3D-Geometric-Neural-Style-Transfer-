SCRIPT_PATH="run_nst.py"

CONTENT_OBJ="preprocessed_meshes/table/table_0sbd.xyzn"

STYLE_OBJ="preprocessed_meshes/table/table_1sbd.xyzn"

LAYERS=(
    '0' '1' '2' '3' '4' '5'       # single layer
)

for idx in "${!LAYERS[@]}"
do
  EXP_NAME="single_$((idx))"

  args=(
      --num_iters 2000
      --save_every 500
      --content_shape $CONTENT_OBJ
      --style_shape $STYLE_OBJ
      --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
      --output_dir "logs/table_layers"
      --style_weight 10
      --num_points 32000
      --layers ${LAYERS[idx]}
      --layers_exp_suffix $EXP_NAME
  )

  python $SCRIPT_PATH "${args[@]}"
done
