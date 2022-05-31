SCRIPT_PATH="run_nst.py"

#experiment 0
args=(
    --num_epochs 300
    --save_every 50
    --content_shape "preprocessed_meshes/chair_table/chair_table_0sbd.xyzn"
    --style_shape "preprocessed_meshes/chair_table/chair_table_1sbd.xyzn"
    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
    --output_dir "logs/chair_table_nst"
    --resolution 128
    --style_weight 5
    --num_points 80000
)

python $SCRIPT_PATH "${args[@]}"
#
##experiment 1
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 20
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w20.ply
#cd ../..
#
#
#
##experiment 2
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 50
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w50.ply
#cd ../..
#
#
#
##experiment 3
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 100
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w100.ply
#cd ../..
#
#
#
#
##experiment 4
#args=(
#    --num_epochs 500
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 10
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w10_500it.ply
#cd ../..
#
#
#
#
##experiment 5
#args=(
#    --num_epochs 1000
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 10
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w10_1000it.ply
#cd ../..
#
#
#
##experiment 6
#args=(
#    --num_epochs 500
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 50
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w50_500it.ply
#cd ../..
#
#
#
##experiment 7
#args=(
#    --num_epochs 1000
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/table1_nst"
#    --resolution 128
#    --style_weight 50
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w50_1000it.ply
#cd ../..
#
#
#
##experiment 9
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 10
#    --num_points 10000
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w10_10kpts.ply
#cd ../..
#
#
#
##experiment 10
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 10
#    --num_points 20000
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w10_20kpts.ply
#cd ../..
#
#
#
##experiment 11
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/table1/table1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/table1/table1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 10
#    --num_points 50000
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/table1_nst
#mv stylization_it_final.ply stylization_it_final_w10_50kpts.ply
#cd ../..
#
#
#
####################
## bag experiments #
####################
#
##experiment 12
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/bag1/bag1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/bag1/bag1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 5
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/bag1_nst
#mv stylization_it_final.ply stylization_bag_w5.ply
#cd ../..
#
#
#
##experiment 13
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/bag1/bag1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/bag1/bag1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 10
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/bag1_nst
#mv stylization_it_final.ply stylization_bag_w10.ply
#cd ../..
#
#
#
##experiment 14
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/bag1/bag1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/bag1/bag1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 20
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/bag1_nst
#mv stylization_it_final.ply stylization_bag_w20.ply
#cd ../..
#
#
#
##experiment 15
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/bag1/bag1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/bag1/bag1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 50
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/bag1_nst
#mv stylization_it_final.ply stylization_bag_w50.ply
#cd ../..
#
#
#
##experiment 16
#args=(
#    --num_epochs 300
#    --save_every 50
#    --content_shape "preprocessed_meshes/bag1/bag1_0sbd.xyzn"
#    --style_shape "preprocessed_meshes/bag1/bag1_1sbd.xyzn"
#    --pretrained_sdf "logs/bag_experiment2/checkpoints/.pth"
#    --pretrained_enc "pretrained_models/shapenet.pointnet.pth.tar"
#    --output_dir "logs/bag1_nst"
#    --resolution 128
#    --style_weight 100
#)
#
#python $SCRIPT_PATH "${args[@]}"
#
#cd logs/bag1_nst
#mv stylization_it_final.ply stylization_bag_w100.ply
#cd ../..
#
