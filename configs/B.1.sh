ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python main.py \
--expriment_name=${ID} \
--epoch_size=2 \
--checkpoint_dir=checkpoints \
--model_type=vae \
--gamma=20.0 \
| tee output/${ID}/stdout.log

sh anim.sh output/${ID}/
