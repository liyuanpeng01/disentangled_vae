ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python main.py \
--expriment_name=${ID} \
--task_type=onecolor \
--epoch_size=10 \
--checkpoint_dir=checkpoints \
--model_type=stn \
| tee output/${ID}/stdout.log

sh anim.sh output/${ID}/
