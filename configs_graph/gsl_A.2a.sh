ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python -u main.py \
--expriment_name=${ID} \
--task_type=onecolor \
--dist_type=customize \
--epoch_size=100 \
--checkpoint_dir=checkpoints \
--model_type=stn \
--rep_regularize \
--alpha=0.1 \
--learning_rate=5e-4 \
--sigmoid_output \
--training=False \
| tee output/${ID}/stdout.log

sh anim.sh output/${ID}/
