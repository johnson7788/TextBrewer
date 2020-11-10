#set hyperparameters
BERT_DIR=/path/to/bert-base-cased
OUTPUT_ROOT_DIR=/path/to/output_root_dir
DATA_ROOT_DIR=/path/to/data_root_dir
trained_teacher_model_1=/path/to/trained_teacher_model_1_file
trained_teacher_model_2=/path/to/trained_teacher_model_2_file
trained_teacher_model_3=/path/to/trained_teacher_model_3_file

STUDENT_CONF_DIR=config/chinese_bert_config_L4t.json

accu=1
ep=10
lr=2
temperature=8
batch_size=24
length=128
# 最终学习率为1/sopt1
sopt1=30
torch_seed=9580

taskname='cosmetics'
NAME=${taskname}_t${temperature}_MTbaseST4tiny_lr${lr}e${ep}_bs${batch_size}
DATA_DIR=${DATA_ROOT_DIR}/MNLI
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u main.multiteacher.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --data_dir  $DATA_DIR \
    --bert_config_file_T $BERT_DIR/bert_config.json \
    --bert_config_file_S $STUDENT_CONF_DIR/bert_config.json \
    --init_checkpoint_S  $BERT_DIR/pytorch_model.bin \
    --tuned_checkpoint_Ts $trained_teacher_model_1 \
                          $trained_teacher_model_2 \
                          $trained_teacher_model_3 \
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
    --temperature ${temperature} \
    --task_name ${taskname} \
    --output_encoded_layers false \
    --output_attention_layers false
