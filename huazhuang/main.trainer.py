import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger("Main")

import json
import scipy
import os,random
import numpy as np
import torch
from utils_glue import output_modes, processors
from pytorch_pretrained_bert.my_modeling import BertConfig
from transformers import ElectraConfig
from modeling import ElectraSPC
from pytorch_pretrained_bert import BertTokenizer
from optimization import BERTAdam
import config
from utils import divide_parameters, load_and_cache_examples
from modeling import BertSPCSimple,BertForGLUESimpleAdaptorTraining

from textbrewer import DistillationConfig, TrainingConfig, BasicTrainer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_glue import compute_metrics
from functools import partial


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1:
        # 本地运行，不用多GPU架构
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def predict(model,eval_datasets,step,args):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir
    results = {}
    #这里我只用一个task，一个datasets
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
        model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            input_ids, input_mask, segment_ids, labels = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            with torch.no_grad():
                logits = model(input_ids, input_mask, segment_ids)
            cpu_logits = logits.detach().cpu()
            for i in range(len(cpu_logits)):
                pred_logits.append(cpu_logits[i].numpy())
                label_ids.append(labels[i])

        pred_logits = np.array(pred_logits)
        label_ids   = np.array(label_ids)

        #获取最大概率的可能性，即分数
        pred_logits_softmax = scipy.special.softmax(pred_logits, axis=1)
        probability = np.max(pred_logits_softmax, axis=1)
        probability = probability.tolist()

        if args.output_mode == "classification":
            preds = np.argmax(pred_logits, axis=1)
        else: # args.output_mode == "regression":
            preds = np.squeeze(pred_logits)
        result = compute_metrics(eval_task, preds, label_ids)
        logger.info(f"task:,{eval_task}")
        logger.info(f"result: {result}")
        results.update(result)
    #把结果json形式保存出来
    output_json = os.path.join(eval_output_dir, "eval_results-%s.json" % eval_task)
    output_json_data = [(pred, prob) for prob, pred in zip(probability, preds.tolist())]
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f)

    output_eval_file = os.path.join(eval_output_dir, "eval_results-%s.txt" % eval_task)
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} task {} *****".format(step, eval_task))
        writer.write("step: %d ****\n " % step)
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    model.train()
    return results

def main():
    #解析参数
    config.parse()
    args = config.args
    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")
    #set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #arguments check
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    #准备任务
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    # eg： MNLI，['contradiction', 'entailment', 'neutral'] --> [“矛盾”，“必然”，“中立”]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Student的配置
    if args.model_architecture == "electra":
        # 从transformers包中导入ElectraConfig, 并加载配置
        bert_config_S = ElectraConfig.from_json_file(args.bert_config_file_S)
        # (args.output_encoded_layers=='true')  --> True, 默认输出隐藏层的状态
        bert_config_S.output_hidden_states = (args.output_encoded_layers == 'true')
        bert_config_S.output_attentions = (args.output_attention_layers=='true')
        # num_labels；类别个数
        bert_config_S.num_labels = num_labels
        assert args.max_seq_length <= bert_config_S.max_position_embeddings
    else:
        bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
        assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #read data
    train_dataset = None
    eval_datasets  = None
    num_train_steps = None
    # electra和bert都使用的bert的 tokenizer方式
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    # 加载数据集, 计算steps
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if args.aux_task_name:
            aux_train_dataset = load_and_cache_examples(args, args.aux_task_name, tokenizer, evaluate=False, is_aux=True)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aux_train_dataset])
        num_train_steps = int(len(train_dataset)/args.train_batch_size) * args.num_train_epochs
        logger.info("训练数据集已加载")
    if args.do_predict:
        eval_datasets = []
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        for eval_task in eval_task_names:
            eval_datasets.append(load_and_cache_examples(args, eval_task, tokenizer, evaluate=True))
        logger.info("预测数据集已加载")


    # Student的配置
    if args.model_architecture == "electra":
        #加载模型配置, 只用student模型，其实这里相当于训练教师模型，只训练一个模型
        model_S = ElectraSPC(bert_config_S)
    else:
        #加载模型配置, 只用student模型，其实这里相当于训练教师模型，只训练一个模型
        model_S = BertSPCSimple(bert_config_S, num_labels=num_labels,args=args)
    #对加载后的student模型的参数进行初始化, 使用student模型预测
    if args.load_model_type=='bert' and args.model_architecture != "electra":
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        if args.only_load_embedding:
            state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.embeddings')}
            missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
            logger.info(f"Missing keys {list(missing_keys)}")
        else:
            state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
            missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
            assert len(missing_keys)==0
        logger.info("Model loaded")
    elif args.load_model_type=='all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S,map_location='cpu')
        model_S.load_state_dict(state_dict_S)
        logger.info("Model loaded")
    elif args.model_architecture == "electra":
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        missing_keys, unexpected_keys = model_S.load_state_dict(state_dict_S,strict=False)
        logger.info(f"missing keys:{missing_keys}")
        logger.info(f"unexpected keys:{unexpected_keys}")
    else:
        logger.info("Model is randomly initialized.")
    #模型move to device
    model_S.to(device)
    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_S = torch.nn.DataParallel(model_S) #,output_device=n_gpu-1)

    if args.do_train:
        #parameters， params是模型的所有参数组成的列表
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("要训练的模型参数量组是,包括decay_group和no_decay_group: %d", len(all_trainable_params))
        # 优化器设置
        optimizer = BERTAdam(all_trainable_params,lr=args.learning_rate,
                             warmup=args.warmup_proportion,t_total=num_train_steps,schedule=args.schedule,
                             s_opt1=args.s_opt1, s_opt2=args.s_opt2, s_opt3=args.s_opt3)

        logger.info("***** 开始训练 *****")
        logger.info("  样本数是 = %d", len(train_dataset))
        logger.info("  前向 batch size = %d", forward_batch_size)
        logger.info("  训练的steps = %d", num_train_steps)

        ########### 训练的配置 ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device)

        #初始化trainer，执行监督训练，而不是蒸馏。它可以把model_S模型训练成为teacher模型
        distiller = BasicTrainer(train_config = train_config,
                                 model = model_S,
                                 adaptor = BertForGLUESimpleAdaptorTraining)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError
        #训练的dataloader
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        #执行callbakc函数，对eval数据集
        callback_func = partial(predict, eval_datasets=eval_datasets, args=args)
        with distiller:
            #开始训练
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_datasets,step=0,args=args)
        print(res)

if __name__ == "__main__":
    main()
