SCRIPT_PATH="run_nst.py"

CONTENT_OBJ="preprocessed_meshes/table/table_0sbd.xyzn"

STYLE_OBJ="preprocessed_meshes/table/table_1sbd.xyzn"

LAYERS=(
    '0 1 2 3 4 5'                 # all features
    '0' '1' '2' '3' '4' '5'       # single layer
    '0 5' '1 5' '2 5' '3 5' '4 5' # (local, global)
)

for idx in "${!LAYERS[@]}"
do
  if [[ idx -eq 0 ]]
  then
      EXP_NAME="all"
  elif [[ idx -ge 1 && idx -le 6 ]]
  then
      EXP_NAME="single_$((idx - 1))"
  else
      EXP_NAME="comb_$((idx - 7))"
  fi

  args=(
      --num_epochs 1
      --save_every 50
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
