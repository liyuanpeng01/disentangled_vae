# from A.0

ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

rm -r output/${ID}
mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python -u transfer_main.py \
--expriment_name=${ID} \
--task_type=onecolor \
--dist_type=customize \
--epoch_size=1 \
--checkpoint_dir=checkpoints \
--model_type=vae \
--gamma=4.0 \
--alpha=0.1 \
--learning_rate=5e-4 \
--load_features \
--source_model=checkpoints/bl_C/source \
--target_model=checkpoints/bl_C/target \
--task_id 2 \
| tee output/${ID}/stdout.log
