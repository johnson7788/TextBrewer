import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

import os, time
import numpy as np
import torch
from utils_data import output_modes, processors
from pytorch_pretrained_bert.my_modeling import BertConfig
from pytorch_pretrained_bert import BertTokenizer
import config
from utils_data import divide_parameters, load_and_cache_examples
from modeling import BertSPCSimple

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_data import compute_metrics


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("输出目录已经存在，且不为空")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("无效的gradient_accumulation_steps参数：{}应>= 1".format(
            args.gradient_accumulation_steps))
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


def predict(model, eval_dataset, args):
    # 任务名字
    eval_task = args.task_name
    eval_output_dir = args.output_dir
    results = {}
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    logger.info("***** 开始预测 *****")
    logger.info(" 任务名称 = %s", eval_task)
    logger.info(" 样本数 = %d", len(eval_dataset))
    logger.info(" Batch size = %d", args.predict_batch_size)
    # 评估样本
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
    model.eval()

    #起始时间
    start_time = time.time()
    #存储预测值
    pred_logits = []
    label_ids = []
    for batch in tqdm(eval_dataloader, desc="评估中", disable=None):
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
    label_ids = np.array(label_ids)

    #找到最大的概率label
    preds = np.argmax(pred_logits, axis=1)
    result = compute_metrics(eval_task, preds, label_ids)
    logger.info(f"task:,{eval_task}")
    logger.info(f"result: {result}")
    results.update(result)

    cost_time = time.time() - start_time
    logger.info(
        f"--- 评估{len(eval_dataset)}条数据的总耗时是 {cost_time} seconds, 每条耗时 {cost_time / len(eval_dataset)} seconds ---")
    #写到日志文件
    output_eval_file = os.path.join(eval_output_dir, "eval_results-%s.txt" % eval_task)
    with open(output_eval_file, "a") as writer:
        logger.info("***** 评估结果 task {} *****".format(eval_task))
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    model.train()
    return results


def main():
    # 解析参数
    config.parse()
    args = config.args
    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")

    # 解析参数, 判断使用的设备
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    # 加载student的配置文件, 校验最大序列长度小于我们的配置中的序列长度
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    # 准备task
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    # 所有的labels
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 读取数据
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    eval_dataset, examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    logger.info("评估数据集已加载")

    model_S = BertSPCSimple(bert_config_S, num_labels=num_labels, args=args)
    # 加载student模型
    assert args.tuned_checkpoint_S is not None
    state_dict_S = torch.load(args.tuned_checkpoint_S, map_location='cpu')
    model_S.load_state_dict(state_dict_S)
    logger.info("Student模型已加载")

    # 开始预测
    res = predict(model_S, eval_dataset, args=args)
    print(res)


if __name__ == "__main__":
    main()
