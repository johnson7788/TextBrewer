[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示MSRA NER(中文命名实体识别)任务上，在**分布式数据并行训练**(Distributed Data-Parallel, DDP)模式(single node, muliti-GPU)下的[Chinese-ELECTRA-base](https://github.com/ymcui/Chinese-ELECTRA)模型蒸馏。


* ner_ElectraTrain_dist.sh : 训练教师模型(ELECTRA-base)。
* ner_ElectraDistill_dist.sh : 将教师模型蒸馏到学生模型(ELECTRA-small)。


运行脚本前，请根据自己的环境设置相应变量：

注意我们仅使用判别器就ok了, 下载好判别器
* ELECTRA_DIR_BASE :  存放Chinese-ELECTRA-base模型的目录，包含vocab.txt，pytorch_model.bin和config.json。

* OUTPUT_DIR : 存放训练好的模型权重文件和日志。
* DATA_DIR : MSRA NER数据集目录，包含
  * msra_train_bio.txt
  * msra_test_bio.txt

对于蒸馏，需要设置:

* ELECTRA_DIR_SMALL :  Chinese-ELECTRA-small预训练权重所在目录。应包含pytorch_model.bin。 也可不提供预训练权重，则学生模型将随机初始化。
* student_config_file : 学生模型配置文件，一般文件名为config.json，也位于 $\{ELECTRA_DIR_SMALL\}。
* trained_teacher_model_file : 在MSRA NER任务上训练好的ELECTRA-base教师模型。

该脚本在 **PyTorch==1.2, Transformers==2.8** 下测试通过。


# 分布式训练
```buildoutcfg
python -m torch.distributed.launch    # 分布式运行
--nproc_per_node=2   # 2个节点上运行  
main.train.dist.py 
--vocab_file electra_model/vocab.txt 
--do_lower_case 
--bert_config_file_T none 
--bert_config_file_S electra_model/config.json 
--init_checkpoint_S electra_model/pytorch_model.bin 
--do_train 
--do_eval 
--do_predict 
--max_seq_length 160 
--train_batch_size 12 
--random_seed 1337 
--train_file data/msra_train_bio.txt 
--predict_file data/msra_test_bio.txt 
--num_train_epochs 2 
--learning_rate 30e-5 
--ckpt_frequency 2 
--official_schedule linear 
--output_dir output_dir 
--gradient_accumulation_steps 1 
--output_encoded_layers true 
--output_attention_layers false 
--lr_decay 0.8
--no_cuda
--local_rank -1
```

# 单机训练
```buildoutcfg
python main.train.dist.py --vocab_file electra_model/vocab.txt --do_lower_case --bert_config_file_T none --bert_config_file_S electra_model/config.json --init_checkpoint_S electra_model/pytorch_model.bin --do_train --do_eval --do_predict --max_seq_length 160 --train_batch_size 12 --random_seed 1337 --train_file data/msra_train_bio.txt --predict_file data/msra_test_bio.txt --num_train_epochs 2 --learning_rate 30e-5 --ckpt_frequency 2 --official_schedule linear --output_dir output_dir --gradient_accumulation_steps 1 --output_encoded_layers true --output_attention_layers false --lr_decay 0.8 --local_rank -1 --no_cuda
```