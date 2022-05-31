SCRIPT_PATH="experiment_scripts/test_sdf.py"

CKPT_PATH="logs/bag_experiment/checkpoints/model_current.pth"
EXP_NAME="experiment_bag_rec"
RESOLUTION=192

args=(
        --checkpoint_path=$CKPT_PATH
        --experiment_name=$EXP_NAME
        --resolution=$RESOLUTION
)


python $SCRIPT_PATH "${args[@]}"