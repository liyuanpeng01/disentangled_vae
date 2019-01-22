# from D.sh

ID=$(basename "$0" | sed "s/.sh$//g")
cd $(dirname $(dirname $(readlink -f $0)))

rm -r output/${ID}
mkdir -p output/${ID}

PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$1 \
python -u main.py \
--expriment_name=${ID} \
--real_data \
--task_type=onecolor \
--dist_type=real_scustomize \
--epoch_size=100 \
--checkpoint_dir=checkpoints \
--model_type=beta \
--gamma=4.0 \
| tee output/${ID}/stdout.log

sh anim.sh output/${ID}/
