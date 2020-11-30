import logging
import time
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger("Main")

import os,random
import numpy as np
import torch
from utils_glue import output_modes, processors
from pytorch_pretrained_bert.my_modeling import BertConfig
from pytorch_pretrained_bert import BertTokenizer
from optimization import BERTAdam
import config
from utils import divide_parameters, load_and_cache_examples
from modeling import BertForGLUESimple,BertForGLUESimpleAdaptor

from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
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

def predict(model,eval_datasets,step,args, examples=None, label_list=None):
    """

    :param model:
    :param eval_datasets:
    :param step:
    :param args:
    :param examples:  样本集合，可以打印前N条样本的预测结果
    :return:
    """
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir
    results = {}
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

        #起始时间
        start_time = time.time()
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
        # 所有真实的labels id
        label_ids = np.array(label_ids)
        if args.output_mode == "classification":
            # 所有预测的label id
            preds = np.argmax(pred_logits, axis=1)
        else: # args.output_mode == "regression":
            preds = np.squeeze(pred_logits)
        result = compute_metrics(eval_task, preds, label_ids)
        #随机取100个样本，查看结果
        if examples and label_list:
            #抽取100个
            num_example = 100
            print(f"随机打印{num_example}个预测结果")
            total_examples = len(label_ids)
            display_examples = random.sample(range(total_examples),num_example)
            print("样本          关键字          真实标签         预测标签")
            for exp_idx in display_examples:
                print('%30s  %10s  %8s  %8s' % (examples[exp_idx].text_a, examples[exp_idx].text_b, examples[exp_idx].label,label_list[preds[exp_idx]]))
        logger.info(f"task:,{eval_task}")
        logger.info(f"result: {result}")
        results.update(result)

    cost_time = time.time() - start_time
    logger.info(f"--- 评估{len(eval_dataset)}条数据的总耗时是 {cost_time} seconds, 每条耗时 {cost_time/len(eval_dataset)} seconds ---")
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
    #parse arguments
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

    #load bert config
    bert_config_T = BertConfig.from_json_file(args.bert_config_file_T)
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_T.max_position_embeddings
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #Prepare GLUE task
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    #read data
    train_dataset = None
    eval_datasets  = None
    num_train_steps = None
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    # 加载数据集
    if args.do_train:
        train_dataset,examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if args.aux_task_name:
            aux_train_dataset,examples = load_and_cache_examples(args, args.aux_task_name, tokenizer, evaluate=False, is_aux=True)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aux_train_dataset])
        num_train_steps = int(len(train_dataset)/args.train_batch_size) * args.num_train_epochs
    if args.do_predict:
        eval_datasets = []
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        for eval_task in eval_task_names:
            eval_dataset, examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
            eval_datasets.append(eval_dataset)
    logger.info("数据集加载成功")

    #加载模型，加载teacher和student模型
    model_T = BertForGLUESimple(bert_config_T, num_labels=num_labels,args=args)
    model_S = BertForGLUESimple(bert_config_S, num_labels=num_labels,args=args)
    #加载teacher模型参数
    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True
    #Load student
    if args.load_model_type=='bert':
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
    else:
        logger.info("Student模型没有可加载参数，随机初始化参数 randomly initialized.")
    model_T.to(device)
    model_S.to(device)

    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_T = torch.nn.DataParallel(model_T) #,output_device=n_gpu-1)
            model_S = torch.nn.DataParallel(model_S) #,output_device=n_gpu-1)

    if args.do_train:
        #parameters
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))
        #优化器配置
        optimizer = BERTAdam(all_trainable_params,lr=args.learning_rate,
                             warmup=args.warmup_proportion,t_total=num_train_steps,schedule=args.schedule,
                             s_opt1=args.s_opt1, s_opt2=args.s_opt2, s_opt3=args.s_opt3)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device)
        # 定义了一些固定的matches配置文件
        from matches import matches
        intermediate_matches = None
        if isinstance(args.matches,(list,tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f"中间层match信息： {intermediate_matches}")
        distill_config = DistillationConfig(
            temperature = args.temperature,
            intermediate_matches=intermediate_matches)

        logger.info(f"训练配置： {train_config}")
        logger.info(f"蒸馏配置： {distill_config}")
        adaptor_T = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask = args.no_inputs_mask)
        adaptor_S = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask = args.no_inputs_mask)
        # 支持中间状态匹配的通用蒸馏模型
        distiller = GeneralDistiller(train_config = train_config,
                                   distill_config = distill_config,
                                   model_T = model_T, model_S = model_S,
                                   adaptor_T = adaptor_T,
                                   adaptor_S = adaptor_S)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        callback_func = partial(predict, eval_datasets=eval_datasets, args=args, examples=examples)
        with distiller:
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_datasets,step=0,args=args, examples=examples, label_list=label_list)
        print (res)

if __name__ == "__main__":
    main()
