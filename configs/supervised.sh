ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

rm -r output/${ID}
mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python supervised.py \
--expriment_name=${ID} \
--task_type=onecolor \
--dist_type=linear \
--epoch_size=1 \
--checkpoint_dir=checkpoints \
--model_type=stn \
--learning_rate=1e-5 \
#--short_training \
#| tee output/${ID}/stdout.log

#sh anim.sh output/${ID}/
