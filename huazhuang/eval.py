import logging
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
from modeling import BertForGLUESimple,BertForGLUESimpleAdaptorTraining

from textbrewer import DistillationConfig, TrainingConfig, BasicTrainer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_glue import compute_metrics
from functools import partial


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("输出目录已经存在，且不为空")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("无效的gradient_accumulation_steps参数：{}应>= 1".format(
                            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("do_train或do_predict中的至少一个必须为True")

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("使用设备 %s 使用的gpu数 %d； 是否分布式 %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def predict(model,eval_datasets,step,args):
    eval_task_names = args.task_name
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

        if args.output_mode == "classification":
            preds = np.argmax(pred_logits, axis=1)
        else: # args.output_mode == "regression":
            preds = np.squeeze(pred_logits)
        result = compute_metrics(eval_task, preds, label_ids)
        logger.info(f"task:,{eval_task}")
        logger.info(f"result: {result}")
        results.update(result)

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
    #设置随机数种子
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #解析参数, 判断使用的设备
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    #加载student的配置文件
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #准备task
    processor = processors[args.task_name]()
    # 是分类还是回归
    args.output_mode = output_modes[args.task_name]
    #所有的labels
    label_list = processor.get_labels()
    num_labels = len(label_list)

    #读取数据
    eval_datasets  = None
    num_train_steps = None
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.do_predict:
        eval_datasets = []
        eval_task_names = args.task_name
        for eval_task in eval_task_names:
            eval_datasets.append(load_and_cache_examples(args, eval_task, tokenizer, evaluate=True))
    logger.info("数据集已加载")


    #加载模型并初始化, 只用student模型，其实这里相当于在MNLI数据上训练教师模型，只训练一个模型
    model_S = BertForGLUESimple(bert_config_S, num_labels=num_labels,args=args)
    #初始化student模型
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
        logger.info("Model is randomly initialized.")
    model_S.to(device)

    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_S = torch.nn.DataParallel(model_S) #,output_device=n_gpu-1)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_datasets,step=0,args=args)
        print (res)

if __name__ == "__main__":
    main()
