import logging


import os,random
import numpy as np
import torch
from utils_ner import read_features, label2id_dict
from utils import divide_parameters
from transformers import ElectraConfig, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, BertTokenizer, get_constant_schedule
import config
from modeling import ElectraForTokenClassification, ElectraForTokenClassificationAdaptorTraining
from textbrewer import DistillationConfig, TrainingConfig,BasicTrainer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from train_eval import predict, ddp_predict

def args_check(logger, args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        logger.info("rank %d device %s n_gpu %d distributed training %r", torch.distributed.get_rank(), device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def main():
    #parse arguments
    config.parse()
    args = config.args

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
        )
    logger = logging.getLogger("Main")
    #arguments check
    device, n_gpu = args_check(logger, args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank != -1:
        logger.warning(f"Process rank: {torch.distributed.get_rank()}, device : {args.device}, n_gpu : {args.n_gpu}, distributed training : {bool(args.local_rank!=-1)}")

    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")
    #set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    #从transformers包中导入ElectraConfig, 并加载配置
    bert_config_S = ElectraConfig.from_json_file(args.bert_config_file_S)
    # (args.output_encoded_layers=='true')  --> True, 默认输出隐藏层的状态
    bert_config_S.output_hidden_states = (args.output_encoded_layers=='true')
    #num_labels；类别个数
    bert_config_S.num_labels = len(label2id_dict)
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #读取数据
    train_examples = None
    train_dataset = None
    eval_examples = None
    eval_dataset = None
    num_train_steps = None
    # 加载Tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.do_train:
        # 返回所有的样本和样本形成的dataset格式, dataset的格式[all_token_ids,all_input_mask,all_label_ids]
        print("开始加载训练集数据")
        train_examples,train_dataset = read_features(args.train_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    if args.do_predict:
        print("开始加载测试集数据")
        eval_examples,eval_dataset = read_features(args.predict_file,tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    if args.local_rank == 0:
        torch.distributed.barrier()
    #Build Model and load checkpoint
    model_S = ElectraForTokenClassification(bert_config_S)
    #加载student模型的参数, 默认是bert类型
    if args.load_model_type=='bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        #state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
        #missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
        missing_keys, unexpected_keys = model_S.load_state_dict(state_dict_S,strict=False)
        logger.info(f"missing keys:{missing_keys}")
        logger.info(f"unexpected keys:{unexpected_keys}")
    elif args.load_model_type=='all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S,map_location='cpu')
        model_S.load_state_dict(state_dict_S)
    else:
        logger.info("Model is randomly initialized.")
    #模型放到device
    model_S.to(device)

    if args.do_train:
        #parameters
        if args.lr_decay is not None:
            # 分类器层的参数 weight, bias
            outputs_params = list(model_S.classifier.named_parameters())
            #拆分出做学习率衰减的参数和不衰减的参数
            outputs_params = divide_parameters(outputs_params, lr = args.learning_rate)

            electra_params = []
            # eg: 12, encoder层共12层
            n_layers = len(model_S.electra.encoder.layer)
            assert n_layers==12
            for i,n in enumerate(reversed(range(n_layers))):
                encoder_params = list(model_S.electra.encoder.layer[n].named_parameters())
                lr = args.learning_rate * args.lr_decay**(i+1)
                electra_params += divide_parameters(encoder_params, lr = lr)
                logger.info(f"{i}:第{n}层的学习率是:{lr}")
            embed_params = [(name,value) for name,value in model_S.electra.named_parameters() if 'embedding' in name]
            logger.info(f"{[name for name,value in embed_params]}")
            lr = args.learning_rate * args.lr_decay**(n_layers+1)
            electra_params += divide_parameters( embed_params, lr = lr)
            logger.info(f"embedding层的学习率 lr:{lr}")
            all_trainable_params = outputs_params + electra_params
            assert sum(map(lambda x:len(x['params']), all_trainable_params))==len(list(model_S.parameters())),\
                (sum(map(lambda x:len(x['params']), all_trainable_params)), len(list(model_S.parameters())))
        else:
            params = list(model_S.named_parameters())
            all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("可训练的参数all_trainable_params共有: %d", len(all_trainable_params))

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        #生成dataloader
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        # 根据epoch计算出运行多少steps
        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(all_trainable_params, lr=args.learning_rate, correct_bias = False)
        if args.official_schedule == 'const':
            # 常数学习率
            scheduler_class = get_constant_schedule_with_warmup
            scheduler_args = {'num_warmup_steps':int(args.warmup_proportion*num_train_steps)}
            #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*num_train_steps))
        elif args.official_schedule == 'linear':
            # 线性学习率
            scheduler_class = get_linear_schedule_with_warmup
            # warmup多少步，10%的step是warmup的
            scheduler_args = {'num_warmup_steps':int(args.warmup_proportion*num_train_steps), 'num_training_steps': num_train_steps}
            #scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.warmup_proportion*num_train_steps), num_training_steps = num_train_steps)
        elif args.official_schedule == 'const_nowarmup':
            scheduler_class = get_constant_schedule
            scheduler_args = {}
        else:
            raise NotImplementedError

        logger.warning("***** 开始 训练 *****")
        logger.warning("local_rank %d 原样本数 = %d",  args.local_rank, len(train_examples))
        logger.warning("local_rank %d split之后的样本数 = %d", args.local_rank, len(train_dataset))
        logger.warning("local_rank %d 前向 batch size = %d", args.local_rank, forward_batch_size)
        logger.warning("local_rank %d 训练的steps = %d", args.local_rank, num_train_steps)

        ########### TRAINING ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,  #保存频率
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device,
            fp16=args.fp16,
            local_rank = args.local_rank)

        logger.info(f"训练的配置文件:")
        logger.info(f"{train_config}")
        # 初始化训练器
        distiller = BasicTrainer(train_config = train_config,
                                   model = model_S,
                                   adaptor = ElectraForTokenClassificationAdaptorTraining)

        # 初始化callback函数，使用的ddp_predict函数进行评估
        callback_func = partial(ddp_predict, 
                eval_examples=eval_examples,
                eval_dataset=eval_dataset,
                args=args)
        with distiller:
            distiller.train(optimizer, scheduler_class=scheduler_class, 
                              scheduler_args=scheduler_args,
                              max_grad_norm = 1.0,
                              dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = ddp_predict(model_S,eval_examples,eval_dataset,step=0,args=args)
        print (res)




if __name__ == "__main__":
    main()
