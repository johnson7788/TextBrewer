[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示MNLI句对分类任务上的蒸馏。GLUE中的其他任务的蒸馏也类似。

* run_mnli_train.sh : 在MNLI数据上训练教师模型(bert-base-cased)
python main.trainer.py --vocab_file bert_model/vocab.txt --data_dir data_root_dir/MNLI --bert_config_file_T none --bert_config_file_S bert_model/config.json --init_checkpoint_S bert_model/pytorch_model.bin --do_lower_case --do_train --do_eval --do_predict --max_seq_length 128 --train_batch_size 24 --random_seed 9580 --num_train_epochs 3 --learning_rate 2e-5 --ckpt_frequency 1 --schedule slanted_triangular --s_opt1 30 --output_dir output_root_dir/mnli_base_lr2e3_bs24_teacher --gradient_accumulation_steps 1 --task_name mnli --output_att_sum false --output_encoded_layers false --output_attention_layers false
模型保存为gs{global_step}.pkl 格式

* run_mnli_distill_T4tiny.sh : 在MNLI上蒸馏教师模型到T4Tiny
python main.distill.py --vocab_file bert_model/vocab.txt --data_dir data_root_dir/MNLI --bert_config_file_T bert_model/config.json --bert_config_file_S ../student_config/bert_base_cased_config/bert_config_L4t.json --tuned_checkpoint_T trained_teacher_model/gs1249.pkl --load_model_type none --do_lower_case --do_train --do_eval --do_predict --max_seq_length 128 --train_batch_size 24 --random_seed 9580 --num_train_epochs 40 --learning_rate 10e-5 --ckpt_frequency 1 --schedule slanted_triangular --s_opt1 30 --output_dir output_root_dir/mnli_t8_TbaseST4tiny_AllSmmdH1_lr10e40_bs24 --gradient_accumulation_steps 1 --temperature 8 --task_name mnli --output_att_sum false --output_encoded_layers true --output_attention_layers false --matches L4t_hidden_mse L4_hidden_smmd

* run_mnli_distill_T4tiny_eval.sh
评估模型  python main.distill.py  --vocab_file final_model/vocab.txt --data_dir data_root_dir/MNLI --bert_config_file_T bert_model/config.json --bert_config_file_S ../student_config/bert_base_cased_config/bert_config_L4t.json --tuned_checkpoint_T trained_teacher_model/gs1249.pkl --load_model_type all --tuned_checkpoint_S final_model/gs3747.pkl --do_eval --do_predict --max_seq_length 128 --train_batch_size 24 --random_seed 9580 --num_train_epochs 40 --learning_rate 10e-5 --ckpt_frequency 1 --schedule slanted_triangular --s_opt1 30 --output_dir output_root_dir/mnli_t8_TbaseST4tiny_AllSmmdH1_lr10e40_bs24 --gradient_accumulation_steps 1 --temperature 8 --task_name mnli --output_att_sum false --output_encoded_layers true --output_attention_layers false --matches L4t_hidden_mse L4_hidden_smmd

* run_mnli_distill_multiteacher.sh : 执行多教师蒸馏，将多个教师模型压缩到一个学生模型

运行脚本前，请根据自己的环境设置相应变量：

* BERT_DIR : 存放BERT-base-cased模型的目录，包含vocab.txt, pytorch_model.bin, bert_config.json
* OUTPUT_ROOT_DIR : 存放训练好的模型和日志
* DATA_ROOT_DIR : 包含MNLI数据集:


下载地址: https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
  * \$\{DATA_ROOT_DIR\}/MNLI/train.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_matched.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_mismatched.tsv
 
* 如果是运行 run_mnli_distill_T4tiny.sh, 还需要指定训练好的教师模型权重文件 trained_teacher_model
* 如果是运行 run_mnli_distill_multiteacher.sh, 需要指定多个训练好的教师模型权重文件 trained_teacher_model_1, trained_teacher_model_2, trained_teacher_model_3
