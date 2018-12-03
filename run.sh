PYTHONPATH=/home/yuanpeng/Work/projects/tf_models/research/transformer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python main.py \
--epoch_size=1 \
--checkpoint_dir=checkpoints \
--model_type=stn \
#2> stf_temp.log
