#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/23 4:56 下午
# @File  : api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
######################################################
# 使用没有蒸馏的模型预测，改造成一个flask api，
# 包括训练接口api和预测接口api
# /api/train
# /api/predict
######################################################

import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

import os, random, time
import re
import glob
import numpy as np
import torch
from pytorch_pretrained_bert.my_modeling import BertConfig
from pytorch_pretrained_bert import BertTokenizer
from modeling import BertSPCSimple, BertForGLUESimpleAdaptorTraining
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from utils import divide_parameters
from textbrewer import DistillationConfig, TrainingConfig, BasicTrainer
from optimization import BERTAdam
from functools import partial
from tqdm import tqdm
from utils_glue import InputExample, convert_examples_to_features
import argparse
import scipy


from flask import Flask, request, jsonify, abort

app = Flask(__name__)


def load_examples(contents, max_seq_length, tokenizer, label_list):
    """
    :param contents:  eg: [('苹果很好用', '苹果')]  或者 [('苹果很好用', '苹果', '积极')]
    :param max_seq_length:
    :param tokenizer:  初始化后的tokenizer
    :param label_list:
    :return:
    """
    examples = []
    for guid, content in enumerate(contents):
        if len(content) == 2:
            #表示只有sentence和aspect关键字，用于预测
            sentence, aspect = content
            examples.append(
                InputExample(guid=guid, text_a=sentence, text_b=aspect))
        elif len(content) == 3:
            # 表示内容是sentence和aspect关键字和label，一般用于训练
            sentence, aspect, label = content
            examples.append(
                InputExample(guid=guid, text_a=sentence, text_b=aspect, label=label))
        else:
            print(f"这条数据有问题，过滤掉: {guid}: {content}")
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                            output_mode="classification",
                                            cls_token_segment_id=0, pad_token_segment_id=0)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


class TorchAsBertModel(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.label_list = ["积极", "消极", "中性"]
        self.num_labels = len(self.label_list)
        # 判断使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # 预测的batch_size大小
        self.train_batch_size = 8
        # 预测的batch_size大小
        self.predict_batch_size = 64
        # 句子左右最大truncate序列长度
        self.left_max_seq_len = 25
        self.right_max_seq_len = 25
        self.aspect_max_seq_len = 30
        self.load_predict_model()
        self.load_train_model()

    def load_train_model(self):
        """
        初始化训练的模型
        :return:
        """
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.output_encoded_layers = True
        args.output_attention_layers = True
        args.output_att_score = True
        args.output_att_sum = True
        self.learning_rate = 2e-05
        #学习率 warmup的比例
        self.warmup_proportion = 0.1
        self.num_train_epochs = 1
        #使用的学习率scheduler
        self.schedule = 'slanted_triangular'
        self.s_opt1 = 30.0
        self.s_opt2 = 0.0
        self.s_opt3 = 1.0
        self.weight_decay_rate = 0.01
        #训练多少epcoh保存一次模型
        self.ckpt_frequency = 1
        #模型和日志保存的位置
        self.output_dir = "output_root_dir/train_api"
        #梯度累积步数
        self.gradient_accumulation_steps = 1
        self.args = args
        # 解析配置文件, 教师模型和student模型的vocab是不变的
        self.vocab_file = "mac_bert_model/vocab.txt"
        self.bert_config_file_S = "mac_bert_model/config.json"
        self.tuned_checkpoint_S = "mac_bert_model/pytorch_model.bin"
        self.max_seq_length = 70
        # 加载student的配置文件, 校验最大序列长度小于我们的配置中的序列长度
        bert_config_S = BertConfig.from_json_file(self.bert_config_file_S)

        # 加载tokenizer
        tokenizer = BertTokenizer(vocab_file=self.vocab_file)

        # 加载模型
        model_S = BertSPCSimple(bert_config_S, num_labels=self.num_labels, args=self.args)
        state_dict_S = torch.load(self.tuned_checkpoint_S, map_location=self.device)
        state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.')}
        missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
        #验证下参数没有丢失
        assert len(missing_keys) == 0
        self.train_tokenizer = tokenizer
        self.train_model = model_S
        logger.info(f"训练模型{self.tuned_checkpoint_S}加载完成")

    def load_predict_model(self, model_file="trained_teacher_model/gs3024.pkl"):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.output_encoded_layers = True
        args.output_attention_layers = True
        args.output_att_score = True
        args.output_att_sum = True
        self.args = args
        # 解析配置文件, 教师模型和student模型的vocab是不变的
        self.vocab_file = "bert_model/vocab.txt"
        # 这里是使用的teacher的config和微调后的teacher模型, 也可以换成student的config和蒸馏后的student模型
        # student config:  config/chinese_bert_config_L4t.json
        # distil student model:  distil_model/gs8316.pkl
        self.bert_config_file_S = "bert_model/config.json"
        self.tuned_checkpoint_S = model_file
        self.max_seq_length = 70
        # 加载student的配置文件, 校验最大序列长度小于我们的配置中的序列长度
        bert_config_S = BertConfig.from_json_file(self.bert_config_file_S)

        # 加载tokenizer
        tokenizer = BertTokenizer(vocab_file=self.vocab_file)

        # 加载模型
        model_S = BertSPCSimple(bert_config_S, num_labels=self.num_labels, args=self.args)
        state_dict_S = torch.load(self.tuned_checkpoint_S, map_location=self.device)
        model_S.load_state_dict(state_dict_S)
        if self.verbose:
            print("模型已加载")
        self.predict_tokenizer = tokenizer
        self.predict_model =  model_S
        logger.info(f"预测模型{model_file}加载完成")

    def truncate(self, input_text, max_len, trun_post='post'):
        """
        实施截断数据
        :param input_text:
        :param max_len:   eg: 15
        :param trun_post: 截取方向，向前还是向后截取，
                        "pre"：截取前面的， "post"：截取后面的
        :return:
        """
        if max_len is not None and len(input_text) > max_len:
            if trun_post == "post":
                return input_text[-max_len:]
            else:
                return input_text[:max_len]
        else:
            return input_text

    def do_truncate_data(self, data):
        """
        对数据做truncate
        :param data:针对不同类型的数据进行不同的截断
        :return:返回列表，是截断后的文本，aspect
        所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
        """
        def aspect_truncate(content, aspect,aspect_start,aspect_end):
            """
            截断函数
            :param content:
            :param aspect:
            :param aspect_start:
            :param aspect_end:
            :return:
            """
            text_left = content[:aspect_start]
            text_right = content[aspect_end:]
            text_left = self.truncate(text_left, self.left_max_seq_len)
            aspect = self.truncate(aspect, self.aspect_max_seq_len)
            text_right = self.truncate(text_right, self.right_max_seq_len, trun_post="pre")
            new_content = text_left + aspect + text_right
            return new_content
        contents = []
        #保存关键字的索引，[(start_idx, end_idx)...]
        locations = []
        for one_data in data:
            if len(one_data) == 2:
                #不带aspect关键字的位置信息，自己查找位置
                content, aspect = one_data
                iter = re.finditer(aspect, content)
                for m in iter:
                    aspect_start, aspect_end = m.span()
                    new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
                    contents.append((new_content, aspect))
                    locations.append((aspect_start,aspect_end))
            elif len(one_data) == 4:
                # 不带label时，长度是4，
                content, aspect, aspect_start, aspect_end = one_data
                new_content = aspect_truncate(content, aspect, aspect_start,aspect_end)
                contents.append((new_content, aspect))
                locations.append((aspect_start, aspect_end))
            elif len(one_data) == 5:
                content, aspect, aspect_start, aspect_end, label = one_data
                new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
                contents.append((new_content, aspect, label))
                locations.append((aspect_start, aspect_end))
            else:
                raise Exception(f"这条数据异常: {one_data},数据长度或者为2, 4，或者为5")
        return contents, locations

    def predict_batch(self, data, truncated=False, model_file=None):
        """
        batch_size数据处理
        :param data: 是一个要处理的数据列表[(content,aspect),...,]
        :param truncated: 是否要截断数据
        :param model_file: 模型文件
        :return:, 返回格式是 [(predicted_label, predict_score),...]
        """
        #如果为None，就不改变加载的模型，否则就改变加载的模型
        if model_file:
            self.load_predict_model(model_file=model_file)
        if truncated:
            data, locations = self.do_truncate_data(data)
        eval_dataset = load_examples(data, self.max_seq_length, self.predict_tokenizer, self.label_list)
        if self.verbose:
            print("评估数据集已加载")

        predictids, probability = self.do_predict(model=self.predict_model, eval_dataset=eval_dataset)
        if self.verbose:
            print(f"预测的结果是: {predictids}, {[self.label_list[id] for id in predictids]}")

        #把id变成标签
        predict_labels = [self.label_list[r] for r in predictids]
        results = list(zip(predict_labels,probability,data,locations))
        return results

    def do_predict(self, model, eval_dataset, step=0):
        """
        :param eval_dataset:
        :param model 参数必须携带，因为训练的callback会调用评估模型时会传入model
        :return: 2个list，一个是预测的id列表，一个是预测的probability列表
        """
        # 任务名字
        if self.verbose:
            print("***** 开始预测 *****")
            print(" 样本数 = %d", len(eval_dataset))
            print(" Step数 = %d", step)
            print(" Batch size = %d", self.predict_batch_size)
        # 评估样本
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.predict_batch_size)
        model.eval()
        model.to(self.device)
        # 起始时间
        start_time = time.time()
        # 存储预测值
        pred_logits = []
        for batch in tqdm(eval_dataloader, desc="评估中", disable=True):
            input_ids, input_mask, segment_ids, _ = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = model(input_ids, input_mask, segment_ids)
            cpu_logits = logits.detach().cpu()
            for i in range(len(cpu_logits)):
                pred_logits.append(cpu_logits[i].numpy())
        pred_logits = np.array(pred_logits)

        # 找到最大的概率label
        preds = np.argmax(pred_logits, axis=1)
        if self.verbose:
            print(f"preds: {preds}")
        predictids = preds.tolist()
        #获取最大概率的可能性，即分数
        pred_logits_softmax = scipy.special.softmax(pred_logits, axis=1)
        probability = np.max(pred_logits_softmax, axis=1)
        probability = probability.tolist()

        cost_time = time.time() - start_time
        if self.verbose:
            print(
                f"--- 评估{len(eval_dataset)}条数据的总耗时是 {cost_time} seconds, 每条耗时 {cost_time / len(eval_dataset)} seconds ---")
        return predictids, probability

    def do_train(self, data, truncated=False):
        """
        训练模型, 数据集分成2部分，训练集和验证集, 默认比例9:1
        :param data: 输入的数据，注意如果做truncated，那么输入的数据为 [(content,aspect,start_idx, end_idx, label),...,]
        :param truncated: 是否要截断，截断按照 self.left_max_seq_len, self.right_max_seq_len进行
        :return:
        """
        if truncated:
            data, locations = self.do_truncate_data(data)
        train_data_len = int(len(data) * 0.9)
        train_data = data[:train_data_len]
        eval_data = data[train_data_len:]
        train_dataset = load_examples(train_data, self.max_seq_length, self.train_tokenizer, self.label_list)
        eval_dataset = load_examples(eval_data, self.max_seq_length, self.train_tokenizer, self.label_list)
        logger.info("训练数据集已加载,开始训练")
        num_train_steps = int(len(train_dataset) / self.train_batch_size) * self.num_train_epochs
        forward_batch_size = int(self.train_batch_size / self.gradient_accumulation_steps)
        # 开始训练
        params = list(self.train_model.named_parameters())
        all_trainable_params = divide_parameters(params, lr=self.learning_rate, weight_decay_rate=self.weight_decay_rate)
        # 优化器设置
        optimizer = BERTAdam(all_trainable_params, lr=self.learning_rate,
                             warmup=self.warmup_proportion, t_total=num_train_steps, schedule=self.schedule,
                             s_opt1=self.s_opt1, s_opt2=self.s_opt2, s_opt3=self.s_opt3)

        logger.info("***** 开始训练 *****")
        logger.info("  训练样本数是 = %d", len(train_dataset))
        logger.info("  评估样本数是 = %d", len(eval_dataset))
        logger.info("  前向 batch size = %d", forward_batch_size)
        logger.info("  训练的steps = %d", num_train_steps)
        ########### 训练的配置 ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            ckpt_frequency = self.ckpt_frequency,
            log_dir = self.output_dir,
            output_dir = self.output_dir,
            device = self.device)
        #初始化trainer，执行监督训练，而不是蒸馏。它可以把model_S模型训练成为teacher模型
        distiller = BasicTrainer(train_config = train_config,
                                 model = self.train_model,
                                 adaptor = BertForGLUESimpleAdaptorTraining)

        train_sampler = RandomSampler(train_dataset)
        #训练的dataloader
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=forward_batch_size, drop_last=True)
        #执行callbakc函数，对eval数据集
        callback_func = partial(self.do_predict, eval_dataset=eval_dataset)
        with distiller:
            #开始训练
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=self.num_train_epochs, callback=callback_func)
        logger.info(f"训练完成")
        return "Done"
@app.route("/api/predict", methods=['POST'])
def predict():
    """
    接收POST请求，获取data参数
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    # model = TorchAsBertModel()
    results = model.predict_batch(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)


@app.route("/api/predict_truncate", methods=['POST'])
def predict_truncate():
    """
    接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    # model = TorchAsBertModel()
    results = model.predict_batch(test_data, truncated=True)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/predict_truncate_model", methods=['POST'])
def predict_truncate_model():
    """
    使用self.output_dir 下的最新的模型的文件
    接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    # model = TorchAsBertModel()
    list_of_files = glob.glob(os.path.join(model.output_dir, "*.pkl"))
    latest_model_file = max(list_of_files, key=os.path.getctime)
    results = model.predict_batch(test_data, truncated=True, model_file=latest_model_file)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)


@app.route("/api/train", methods=['POST'])
def train():
    """
    接收data参数，
    Args:
        data: 训练的数据，是一个文字列表, [(content,aspect,label),...,]
    Returns:
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{data}, 进行训练")
    # model = TorchAsBertModel()
    results = model.do_train(data)
    return jsonify(results)


@app.route("/api/train_truncate", methods=['POST'])
def train_truncate():
    """
    接收data参数，data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        data: 训练的数据，是一个文字列表, [(content,aspect,start_idx, end_idx, label),...,]
    Returns:
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{data}, 进行训练")
    # model = TorchAsBertModel()
    results = model.do_train(data, truncated=True)
    return jsonify(results)



if __name__ == "__main__":
    model = TorchAsBertModel()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
