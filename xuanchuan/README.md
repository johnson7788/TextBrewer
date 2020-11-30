# 数据准备
使用utils下的cosmetics_data_utils.py准备数据, 数据有一个简单示例

# 步骤1
* run_train.sh : 训练教师模型(bert-base-cased)
python main.trainer.py --vocab_file bert_model/vocab.txt --data_dir data_root_dir/caiye --bert_config_file_T none --bert_config_file_S bert_model/config.json --init_checkpoint_S bert_model/pytorch_model.bin --do_lower_case --do_train --do_predict --max_seq_length 20 --train_batch_size 24 --random_seed 9580 --num_train_epochs 6 --learning_rate 2e-5 --ckpt_frequency 1 --schedule slanted_triangular --s_opt1 30 --output_dir output_root_dir/caiye --gradient_accumulation_steps 1 --task_name caiye --output_att_sum false --output_encoded_layers false --output_attention_layers false
模型保存为gs{global_step}.pkl 格式

# 步骤2
* run_distill_T4tiny.sh : 蒸馏教师模型到T4Tiny
python main.distill.py --vocab_file bert_model/vocab.txt --data_dir data_root_dir/caiye --bert_config_file_T bert_model/config.json --bert_config_file_S config/chinese_bert_config_L4t.json --tuned_checkpoint_T trained_teacher_model/gs3024.pkl --load_model_type none --do_lower_case --do_train --do_predict --max_seq_length 70 --train_batch_size 24 --random_seed 9580 --num_train_epochs 40 --learning_rate 10e-5 --ckpt_frequency 1 --schedule slanted_triangular --s_opt1 30 --output_dir output_root_dir/t8_TbaseST4tiny_AllSmmdH1_lr10e40_bs24 --gradient_accumulation_steps 1 --temperature 8 --task_name cosmetics --output_att_sum false --output_encoded_layers true --output_attention_layers false --matches L4t_hidden_mse L4_hidden_smmd

# 步骤3
* run_distill_T4tiny_eval.sh
评估模型  python main.distill.py  --vocab_file bert_model/vocab.txt --data_dir data_root_dir/caiye --bert_config_file_T bert_model/config.json --bert_config_file_S config/chinese_bert_config_L4t.json --tuned_checkpoint_T trained_teacher_model/gs3024.pkl --load_model_type all --tuned_checkpoint_S distil_model/gs8316.pkl  --do_predict --max_seq_length 70  --random_seed 9580 --output_dir output_root_dir/t8_TbaseST4tiny_eval  --temperature 8 --task_name cosmetics

# 运行参数介绍
```
--vocab_file VOCAB_FILE   student和teacher训练BERT模型的单词表文件。
--output_dir OUTPUT_DIR   模型checkpoint的输出目录。
 [--data_dir DATA_DIR] 数据目录，包含数据集
 [--do_lower_case]   是否小写输入文本。对于uncased的模型应为True，对于cased的模型应为False。
 [--max_seq_length MAX_SEQ_LENGTH] WordPiece标签化之后的最大总输入序列长度。长度大于此长度的序列将被截断，小于长度的序列将被填充。
 [--do_train]  是否训练
 [--do_predict] 是否在评估集上评估
 [--train_batch_size TRAIN_BATCH_SIZE]  训练集batch_size
 [--predict_batch_size PREDICT_BATCH_SIZE]  预测时batch_size
 [--learning_rate LEARNING_RATE]  初始化的Adam学习率
 [--num_train_epochs NUM_TRAIN_EPOCHS]  总共训练的epoch数目
 [--warmup_proportion WARMUP_PROPORTION] 学习率线性warmup比例E.g., 0.1 = 10% of training.
 [--verbose_logging]  如果为true，则将打印与数据处理有关的所有警告
 [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]  在执行向后/更新过程之前要累积的更新步骤数。
 [--local_rank LOCAL_RANK] local_rank用于在GPU上进行分布式训练
 [--fp16]   是否使用16位浮点精度而不是32位
 [--random_seed RANDOM_SEED]  随机数种子
 [--load_model_type {bert,all,none}] bert表示使用init_checkpoint_S指定的模型加载student模型，用于训练阶段, all表示使用蒸馏好的tuned_checkpoint_S模型加载student，用于评估阶段，none表示随机初始化student
 [--weight_decay_rate WEIGHT_DECAY_RATE] 
 [--PRINT_EVERY PRINT_EVERY]  训练时每多少step打印一次日志
 [--ckpt_frequency CKPT_FREQUENCY]  模型的保存频率
 [--tuned_checkpoint_T TUNED_CHECKPOINT_T] 训练好的teacher模型
 [--tuned_checkpoint_Ts [TUNED_CHECKPOINT_TS [TUNED_CHECKPOINT_TS ...]]]   训练好的多个teacher模型，用于多teacher蒸馏到一个student
 [--tuned_checkpoint_S TUNED_CHECKPOINT_S]  蒸馏好的student模型
 [--init_checkpoint_S INIT_CHECKPOINT_S] student模型的初始参数checkpoint, 可以在继续训练时加载以前的相同结果的student的模型参数
 --bert_config_file_T BERT_CONFIG_FILE_T 教师模型的配置文件
 --bert_config_file_S BERT_CONFIG_FILE_S 学生模型的配置文件
 [--temperature TEMPERATURE] 训练时的温度
 [--teacher_cached]  cached加快运行速度
 [--s_opt1 S_OPT1] [--s_opt2 S_OPT2] [--s_opt3 S_OPT3]
 [--schedule SCHEDULE] [--no_inputs_mask] [--no_logits]
 [--output_att_score {true,false}]
 [--output_att_sum {true,false}]
 [--output_encoded_layers {true,false}]
 [--output_attention_layers {true,false}]
 [--matches [MATCHES [MATCHES ...]]] 使用matches.py中matches字典预先定义的哪些match格式
 [--task_name {caiye}]  任务的名称
 [--aux_task_name {caiye}]
 [--aux_data_dir AUX_DATA_DIR] 
 [--only_load_embedding]
```

# 评估测试
相比于蒸馏前，使用蒸馏的方法，将模型整理到4层transformer结构,模型大小由390MB下降到43MB，大小下降9倍, 在同样的GPU下，推理速度提升到0.8ms每条，速度提升大概6倍左右, 蒸馏后准确率由84%下降到82%。
```buildoutcfg
2020/11/10 02:31:17 - INFO - Main -  result: {'acc': 0.8214285714285714}
2020/11/10 02:31:17 - INFO - Main -  --- 评估1008条数据的总耗时是 0.8449249267578125 seconds, 每条耗时 0.0008382191733708457 seconds ---
2020/11/10 02:31:17 - INFO - Main -  ***** Eval results 0 task cosmetics *****
2020/11/10 02:31:17 - INFO - Main -  acc = 0.8214285714285714
```