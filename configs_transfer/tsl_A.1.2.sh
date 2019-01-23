ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python -u transfer_main.py \
--expriment_name=${ID} \
--task_type=onecolor \
--dist_type=customize \
--epoch_size=1 \
--checkpoint_dir=checkpoints \
--model_type=stn \
--learning_rate=5e-4 \
--sigmoid_output \
--alpha=0.1 \
--load_features \
--source_model=checkpoints/sl_A.1/source \
--target_model=checkpoints/sl_A.1/target \
--task_id 2 \
| tee output/${ID}/stdout.log
