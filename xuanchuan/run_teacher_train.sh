export PYTHONPATH="${PYTHONPATH}:../src"
# 训练
BERT_DIR=bert_model
OUTPUT_ROOT_DIR=output_root_dir
DATA_ROOT_DIR=data_root_dir

accu=1
ep=3
lr=2
batch_size=8
length=70
# 最终学习率为1/sopt1
sopt1=30
torch_seed=9580

taskname='caiye'
NAME=${taskname}_base_lr${lr}e${ep}_bs${batch_size}_teacher
DATA_DIR=${DATA_ROOT_DIR}/caiye
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u main.trainer.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --data_dir  $DATA_DIR \
    --bert_config_file_T none \
    --bert_config_file_S $BERT_DIR/config.json \
    --init_checkpoint_S  $BERT_DIR/pytorch_model.bin \
    --do_train \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 ${sopt1} \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --task_name ${taskname} \
    --output_att_sum false  \
    --output_encoded_layers false \
    --output_attention_layers false
