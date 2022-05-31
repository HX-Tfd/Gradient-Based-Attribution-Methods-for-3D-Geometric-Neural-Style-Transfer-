SCRIPT_PATH="run_nst.py"

CONTENT_OBJ="preprocessed_meshes/table_comparison/table_comparison_content_sbd.xyzn"

STYLE_OBJS=(
  "preprocessed_meshes/table_comparison/table_comparison_guitar_sbd.xyzn"
  "preprocessed_meshes/table_comparison/table_comparison_lamp_sbd.xyzn"
  "preprocessed_meshes/table_comparison/table_comparison_sofa_sbd.xyzn"
  "preprocessed_meshes/table_comparison/table_comparison_table_sbd.xyzn"

)

for i in "${!STYLE_OBJS[@]}"
do
  args=(
    --num_epochs 2000
    --save_every 50
    --content_shape $CONTENT_OBJ
    --style_shape "${STYLE_OBJS[i]}"
    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth" #not used
    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
    --output_dir "logs/table_comparison_nst"
    --style_weight 10
  )
  python $SCRIPT_PATH "${args[@]}"
  cd logs/table_comparison_nst
  mv stylization_it_final.ply stylization_$i.ply
  cd ../..

done
