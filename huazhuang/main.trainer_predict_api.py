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
from modeling import BertSPCSimple, BertForGLUESimpleAdaptorTraining

from textbrewer import DistillationConfig, TrainingConfig, BasicTrainer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_glue import compute_metrics
from functools import partial

from flask import Flask, request, jsonify, abort

######################################################
# 使用没有蒸馏的模型预测，改造成一个flask api，
######################################################

app = Flask(__name__)


def load_and_cache_examples(contents, max_seq_length, tokenizer, label_list):
    """
    :param contents:  eg: [('苹果很好用', '苹果')]
    :param max_seq_length:
    :param tokenizer:  初始化后的tokenizer
    :param label_list:
    :return:
    """
    examples = []
    for guid, content in enumerate(contents):
        sentence, aspect = content
        examples.append(
            InputExample(guid=guid, text_a=sentence, text_b=aspect))
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode="classification",
                                            cls_token_segment_id=0,pad_token_segment_id=0)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


class TorchAsBertModel(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.label_list = ["NEG", "NEU", "POS"]
        self.num_labels = len(self.label_list)
        # 判断使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.tokenizer, self.model = self.load_model()
        #句子左右最大truncate序列长度
        self.left_max_seq_len = 15
        self.right_max_seq_len = 20
        self.aspect_max_seq_len = 30

    def load_model(self):
        #解析配置文件
        self.vocab_file = "config/chinese_bert_config_L4t.json"
        #这里是使用的teacher的config和微调后的teacher模型, 也可以换成student的config和蒸馏后的student模型
        # student config:  config/chinese_bert_config_L4t.json
        # distil student model:  distil_model/gs8316.pkl
        self.bert_config_file_S = "bert_model/config.json"
        self.tuned_checkpoint_S = "trained_teacher_model/gs3024.pkl"
        self.max_seq_length = 70
        # 预测的batch_size大小
        self.predict_batch_size = 64
        # 加载student的配置文件, 校验最大序列长度小于我们的配置中的序列长度
        bert_config_S = BertConfig.from_json_file(self.bert_config_file_S)

        # 加载tokenizer
        tokenizer = BertTokenizer(vocab_file=self.vocab_file)

        # 加载模型
        model_S = BertSPCSimple(bert_config_S, num_labels=self.num_labels)
        state_dict_S = torch.load(self.tuned_checkpoint_S, map_location=self.device)
        model_S.load_state_dict(state_dict_S)
        if self.verbose:
            print("模型已加载")

        return tokenizer, model_S

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

    def clean(self, text_left, aspect, text_right):
        """
        截断数据
        :param text_left:
        :param aspect:
        :param text_right:
        :return:
        """
        text_left = self.truncate(text_left, self.left_max_seq_len)
        aspect = self.truncate(aspect, self.aspect_max_seq_len)
        text_right = self.truncate(text_right, self.right_max_seq_len, trun_post="pre")

        return text_left, aspect, text_right

    def predict_batch(self, data):
        """
        batch_size数据处理
        :param data: 是一个要处理的数据列表
        :return:
        """
        contents = []
        for one_data in data:
            content, aspect, aspect_start, aspect_end = one_data
            text_left = content[:aspect_start]
            text_right = content[aspect_end:]
            text_left, aspect, text_right = self.clean(text_left, aspect, text_right)
            new_content = text_left + aspect + text_right
            contents.append((new_content, aspect))

        eval_dataset = load_and_cache_examples(contents, self.max_seq_length, self.tokenizer, self.label_list)
        if self.verbose:
            print("评估数据集已加载")

        res = self.do_predict(self.model, eval_dataset)
        if self.verbose:
            print(f"预测的结果是: {res}, {[self.label_list[id] for id in res]}")

        # TODO 输入为一条数据，返回也只返回一条结果即可以了
        return res

    def do_predict(self, model, eval_dataset):
        # 任务名字
        results = []
        if self.verbose:
            print("***** 开始预测 *****")
            print(" 样本数 = %d", len(eval_dataset))
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
            input_ids, input_mask, segment_ids = batch
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
        results.extend(preds.tolist())

        cost_time = time.time() - start_time
        if self.verbose:
            print(
            f"--- 评估{len(eval_dataset)}条数据的总耗时是 {cost_time} seconds, 每条耗时 {cost_time / len(eval_dataset)} seconds ---")
        return results

@app.route("/api", methods=['POST'])
def api():
    """
    Args:
        test_data: 需要预测的数据，是一个文字列表
    Returns:
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data',None)
    model = TorchAsBertModel()
    results = model.predict_batch(test_data)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)