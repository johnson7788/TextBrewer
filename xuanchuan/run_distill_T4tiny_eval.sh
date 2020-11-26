export PYTHONPATH="${PYTHONPATH}:../src"

#set hyperparameters
BERT_DIR=bert_model
OUTPUT_ROOT_DIR=output_root_dir
DATA_ROOT_DIR=data_root_dir
FINAL_MODEL=distil_model/gs8316.pkl

STUDENT_CONF=config/chinese_bert_config_L4t.json

lr=10
temperature=8
batch_size=8
length=70
sopt1=30
torch_seed=9580

taskname='cosmetics'
NAME=${taskname}_t${temperature}_TbaseST4tiny_AllSmmdH1_lr${lr}e${ep}_bs${batch_size}
DATA_DIR=${DATA_ROOT_DIR}/cosmetics
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR


python -u main.distill.py \
    --vocab_file $BERT_DIR/vocab.txt \
    --data_dir  $DATA_DIR \
    --bert_config_file_T $BERT_DIR/config.json \
    --bert_config_file_S $STUDENT_CONF \
    --load_model_type all \
    --tuned_checkpoint_S $FINAL_MODEL \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 1 \
    --schedule slanted_triangular \
    --s_opt1 ${sopt1} \
    --output_dir $OUTPUT_DIR \
    --temperature ${temperature} \
    --task_name ${taskname} \
    #--init_checkpoint_S  $BERT_DIR_BASE/pytorch_model_hf.bin \
