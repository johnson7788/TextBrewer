# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

import csv
import logging
import os
import sys
import json
from io import open
import re

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self):
        # 定义一个默认序列长度，如果需要，需要时才用，数据处理时阶段数据，和训练预测时的传入的args的max_seq_length是尽量保持一致的
        max_seq_length = 128
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    @classmethod
    def _read_json(cls, input_file):
        """
        读取json文件
        :param input_file:
        :return:
        """
        with open(input_file, 'r') as f:
            lines = json.load(f)
        return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """MNLI的labels"""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class CosmeticsProcessor(MnliProcessor):
    """处理Cosmetics数据"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "train.json")), set_type="train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "dev.json")),
            set_type = "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "test.json")),
            set_type = "test")
    def get_labels(self):
        """cosmetics的labels"""
        return ["消极","中性","积极"]
    def truncate_sepcial(self, content, keyword, start_idx, end_idx, add_special=True, max_seq_length=70, verbose=False,
                            special='_'):
        """
        截断函数, 按句子截断，保留完整句子，同时满足最大的序列长度
        :param content: 最大长度
        :param keyword:关键字
        :param start_idx: 关键字的起始位置
        :param end_idx: 结束位置
        :param max_seq_length: 最大序列长度
        :param verbose: 是否打印日志
        :param special: 特殊字符，添加在keyword两侧
        :return: newcontent, newkeyword, left_text, right_text
        """
        import re
        if verbose:
            print()
            print("收到的样本长度和内容")
            print(len(content), content)
            print("收到的样本关键字和位置信息")
            print(keyword, start_idx, end_idx)
        # 因为我们的结构是 CLS_texta_+SEP+keyword+SEP
        max_texta_length = max_seq_length - len(keyword) - 3
        # 我们尝试在原句子中MASK掉关键字,但如果句子长度大于我们模型中的最大长度，我们需要截断
        if content[start_idx:end_idx] != keyword:
            print(f"关键字的位置信息不准确: {content}: {keyword}:  {start_idx}:{end_idx}, 获取的关键字是 {content[start_idx:end_idx]}")
            return None, None, None, None
        if add_special:
            # 计算下texta的真正的应该保留的长度，-3是减去CLS,SEP,SEP，这3个special的token
            texta_list = list(content)
            texta_list.insert(start_idx, special)
            texta_list.insert(end_idx + len(special), special)
            texta_special = ''.join(texta_list)
            special_keyword = special + keyword + special
        else:
            special_keyword = keyword
            texta_special = content

        left_text = texta_special[:start_idx]
        if add_special:
            right_text = texta_special[end_idx + len(special) * 2:]
        else:
            right_text = texta_special[end_idx:]
        # 开始检查长度, 如果长度大于最大序列长度，我们要截取关键字上下的句子，直到满足最大长度以内,截取时用句子分隔的方式截取
        if len(texta_special) > max_texta_length:
            # 需要对texta进行阶段,采取怎样的截断方式更合适,按逗号和句号和冒号分隔
            texta_split = re.split('([：。；，！])', texta_special)
            # 确定keyword在列表中的第几个元素中
            special_keyword_idx_list = []
            for t_idx, t in enumerate(texta_split):
                if special_keyword in t:
                    special_keyword_idx_list.append(t_idx)
            key_symbol = False
            # 和start_idx比较，确定是哪个keyword在列表哪个元素中
            if len(special_keyword_idx_list) > 1:
                for kidx in special_keyword_idx_list:
                    before_len = sum(len(i) for i in texta_split[:kidx])
                    all_idx = [m.start() for m in re.finditer(special_keyword, texta_split[kidx])]
                    for before_spe_len in all_idx:
                        if before_len + before_spe_len == start_idx:
                            special_keyword_idx = kidx
                            break
            elif len(special_keyword_idx_list) == 1:
                special_keyword_idx = special_keyword_idx_list[0]
            else:
                # 没有找到关键字，切分之后, 如果关键字中有，。：,那么也是有问题的，只好强制截断
                key_symbol = True

            if not key_symbol:
                # 先从距离special_keyword_idx最远的地方的句子开始去掉，直到满足序列长度小于max_seq_length,所以要减去元素个数max_texta_length +1
                while len(texta_split) > 1 and sum(len(t) for t in texta_split) > (
                        max_texta_length + 1):
                    # 选择从列表的左面弹出句子还是，右面弹出句子, 极端情况是只有2个句子，special_keyword_idx在第一个句子中，那么应该弹出第二个句子
                    if len(texta_split) / 2 - special_keyword_idx > 0:
                        # special_keyword_idx在列表的左半部分，应该从后面弹出句子
                        droptext = texta_split.pop()
                    else:
                        # 从列表的开头弹出句子, 从左侧弹出的话，索引的位置也需要减去1
                        special_keyword_idx -= 1
                        droptext = texta_split.pop(0)
                        start_idx = start_idx - len(droptext)
                        end_idx = end_idx - len(droptext)
            if (len(texta_split) == 1 and len(texta_split[0]) > max_texta_length) or key_symbol:
                # 如果左侧长度大于max_texta_length的一半，那么截断
                keep_length = int((max_texta_length - len(keyword)) / 2)
                if len(left_text) > keep_length:
                    left_text = left_text[-keep_length:]
                if len(right_text) > keep_length:
                    right_text = right_text[:keep_length]
                text_a = left_text + special_keyword + right_text
            else:
                text_a = ''.join(texta_split)
                left_text = text_a[:start_idx]
                right_text = text_a[end_idx:]
        else:
            text_a = texta_special
        if verbose:
            print("处理后的结果，样本的长度和内容是")
            print(len(text_a), text_a)
            print("左侧的样本的长度和内容是")
            print(len(left_text), left_text)
            print("右侧的样本的长度和内容是")
            print(len(right_text), right_text)
            print()
        if special_keyword not in text_a:
            raise Exception("处理后结果不含关键字了")
        if (len(left_text) + len(special_keyword) + len(right_text)) > max_seq_length:
            raise Exception("处理后总长度序列长度太长")
        if len(text_a) > max_seq_length:
            raise Exception("处理后text_a序列长度太长")
        if not add_special and text_a not in content:
            raise Exception("处理完成后文本不在content中了")
        return text_a, special_keyword, left_text, right_text

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.这里，我们需要max_seq_length"""
        examples = []
        #把句子中的关键字围起来,
        for idx, line in enumerate(lines):
            texta, keyword, start_idx, end_idx, label, _, _ = line
            text_a, special_keyword, left_text, right_text =  self.truncate_sepcial(texta,keyword,start_idx,end_idx,add_special=True,max_seq_length=self.max_seq_length,verbose=False)
            if text_a is None:
                #对于位置不准确的字符，只能跳过了，给与默认texta
                text_a = texta
            guid = "%s-%s" % (set_type, idx)
            text_b = keyword
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
class NewcosProcessor(MnliProcessor):
    """处理Newcos数据"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "train.json")), set_type="train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "dev.json")),
            set_type = "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "test.json")),
            set_type = "test")
    def get_labels(self):
        """cosmetics的labels"""
        return ["消极","中性","积极"]
    def _create_examples(self, lines, set_type):
        """处理label-studio收到的数据"""
        examples = []
        for idx, line in enumerate(lines):
            guid = "%s-%s" % (set_type, idx)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class NewComponentsProcessor(MnliProcessor):
    """处理判断是否是成分的数据"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "train.json")), set_type="train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "dev.json")),
            set_type = "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            lines = self._read_json(os.path.join(data_dir, "test.json")),
            set_type = "test")
    def get_labels(self):
        """cosmetics的labels"""
        return ["是","否"]
    def truncate_sepcial(self, content, keyword, start_idx, end_idx, add_special=True, max_seq_length=70, verbose=False,
                            special='_'):
        """
        截断函数, 按句子截断，保留完整句子，同时满足最大的序列长度
        :param content: 最大长度
        :param keyword:关键字
        :param start_idx: 关键字的起始位置
        :param end_idx: 结束位置
        :param max_seq_length: 最大序列长度
        :param verbose: 是否打印日志
        :param special: 特殊字符，添加在keyword两侧
        :return: newcontent, newkeyword, left_text, right_text
        """
        import re
        if verbose:
            print()
            print("收到的样本长度和内容")
            print(len(content), content)
            print("收到的样本关键字和位置信息")
            print(keyword, start_idx, end_idx)
        # 因为我们的结构是 CLS_texta_+SEP+keyword+SEP
        max_texta_length = max_seq_length - len(keyword) - 3
        # 我们尝试在原句子中MASK掉关键字,但如果句子长度大于我们模型中的最大长度，我们需要截断
        if content[start_idx:end_idx] != keyword:
            print(f"关键字的位置信息不准确: {content}: {keyword}:  {start_idx}:{end_idx}, 获取的关键字是 {content[start_idx:end_idx]}")
            return None, None, None, None
        if add_special:
            # 计算下texta的真正的应该保留的长度，-3是减去CLS,SEP,SEP，这3个special的token
            texta_list = list(content)
            texta_list.insert(start_idx, special)
            texta_list.insert(end_idx + len(special), special)
            texta_special = ''.join(texta_list)
            special_keyword = special + keyword + special
        else:
            special_keyword = keyword
            texta_special = content

        left_text = texta_special[:start_idx]
        if add_special:
            right_text = texta_special[end_idx + len(special) * 2:]
        else:
            right_text = texta_special[end_idx:]
        # 开始检查长度, 如果长度大于最大序列长度，我们要截取关键字上下的句子，直到满足最大长度以内,截取时用句子分隔的方式截取
        if len(texta_special) > max_texta_length:
            # 需要对texta进行阶段,采取怎样的截断方式更合适,按逗号和句号和冒号分隔
            texta_split = re.split('([：。；，!])', texta_special)
            # 确定keyword在列表中的第几个元素中
            special_keyword_idx_list = []
            for t_idx, t in enumerate(texta_split):
                if special_keyword in t:
                    special_keyword_idx_list.append(t_idx)
            key_symbol = False
            # 和start_idx比较，确定是哪个keyword在列表哪个元素中
            if len(special_keyword_idx_list) > 1:
                for kidx in special_keyword_idx_list:
                    before_len = sum(len(i) for i in texta_split[:kidx])
                    all_idx = [m.start() for m in re.finditer(special_keyword, texta_split[kidx])]
                    for before_spe_len in all_idx:
                        if before_len + before_spe_len == start_idx:
                            special_keyword_idx = kidx
                            break
            elif len(special_keyword_idx_list) == 1:
                special_keyword_idx = special_keyword_idx_list[0]
            else:
                # 没有找到关键字，切分之后, 如果关键字中有，。：,那么也是有问题的，只好强制截断
                key_symbol = True

            if not key_symbol:
                # 先从距离special_keyword_idx最远的地方的句子开始去掉，直到满足序列长度小于max_seq_length,所以要减去元素个数max_texta_length +1
                while len(texta_split) > 1 and sum(len(t) for t in texta_split) > (
                        max_texta_length + 1):
                    # 选择从列表的左面弹出句子还是，右面弹出句子, 极端情况是只有2个句子，special_keyword_idx在第一个句子中，那么应该弹出第二个句子
                    if len(texta_split) / 2 - special_keyword_idx > 0:
                        # special_keyword_idx在列表的左半部分，应该从后面弹出句子
                        droptext = texta_split.pop()
                    else:
                        # 从列表的开头弹出句子, 从左侧弹出的话，索引的位置也需要减去1
                        special_keyword_idx -= 1
                        droptext = texta_split.pop(0)
                        start_idx = start_idx - len(droptext)
                        end_idx = end_idx - len(droptext)
            if (len(texta_split) == 1 and len(texta_split[0]) > max_texta_length) or key_symbol:
                # 如果左侧长度大于max_texta_length的一半，那么截断
                keep_length = int((max_texta_length - len(keyword)) / 2)
                if len(left_text) > keep_length:
                    left_text = left_text[-keep_length:]
                if len(right_text) > keep_length:
                    right_text = right_text[:keep_length]
                text_a = left_text + special_keyword + right_text
            else:
                text_a = ''.join(texta_split)
                left_text = text_a[:start_idx]
                right_text = text_a[end_idx:]
        else:
            text_a = texta_special
        if verbose:
            print("处理后的结果，样本的长度和内容是")
            print(len(text_a), text_a)
            print("左侧的样本的长度和内容是")
            print(len(left_text), left_text)
            print("右侧的样本的长度和内容是")
            print(len(right_text), right_text)
            print()
        if special_keyword not in text_a:
            raise Exception("处理后结果不含关键字了")
        if (len(left_text) + len(special_keyword) + len(right_text)) > max_seq_length:
            raise Exception("处理后总长度序列长度太长")
        if len(text_a) > max_seq_length:
            raise Exception("处理后text_a序列长度太长")
        if not add_special and text_a not in content:
            raise Exception("处理完成后文本不在content中了")
        return text_a, special_keyword, left_text, right_text
    def _create_examples(self, lines, set_type):
        """处理label-studio收到的数据"""
        examples = []
        #把句子中的关键字围起来,
        for idx, line in enumerate(lines):
            texta, keyword, start_idx, end_idx, label, _, _ = line
            text_a, special_keyword, left_text, right_text =  self.truncate_sepcial(texta,keyword,start_idx,end_idx,add_special=True,max_seq_length=self.max_seq_length,verbose=False)
            if text_a is None:
                #对于位置不准确的字符，只能跳过了，给与默认texta
                text_a = texta
            guid = "%s-%s" % (set_type, idx)
            text_b = keyword
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    将数据文件加载到`InputBatch`
    :param examples:  样本集
    :param label_list:  labels eg： ['contradiction', 'entailment', 'neutral']
    :param max_seq_length:  最大序列长度，eg： 128
    :param tokenizer:  初始化后的tokenizer
    :param output_mode:  例如'classification' 或者 regression
    :param cls_token_at_end: 定义CLStoken的位置：
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    :param pad_on_left: default: False 从左开始pad
    :param cls_token:  default: [CLS]
    :param sep_token:  default: [se[]
    :param pad_token:  default: 0
    :param sequence_a_segment_id:    句子a的id 0
    :param sequence_b_segment_id:    句子b的id1
    :param cls_token_segment_id:  定义与CLS token关联的segment ID (0 for BERT, 2 for XLNet)
    :param pad_token_segment_id:  使用padding的token segment id
    :param mask_padding_with_zero:  mask也padding为0
    :return:
    """
    # label映射为id
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # 每10000条，记录下日志
        if ex_index % 10000 == 0:
            logger.info("完成样本数 %d of %d" % (ex_index, len(examples)))
        #对句子a进行tokenizer
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        # 如果句子b存在，对句子btokenizer
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # in_place修改`tokens_a`和`tokens_b`以便总长度小于指定长度
            # 因为 [CLS], [SEP], [SEP] 的存在，所以总长度需要减去3 "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # 如果不存在句子b，那么总长度减去2即可。 Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # Bert的惯例是:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # 其中，“type_ids”用于指示这是第一个序列还是第二个序列。在预训练期间学习了用于`type=0` and `type=1` 的嵌入向量，并将其添加到单词嵌入向量(和位置向量)中。 不是必须的
        # 因为[SEP]token明确地分隔了序列，但是使模型更容易学习序列的概念。
        # 对于分类任务，第一个向量(对应于[CLS])用作“句子向量”。请注意，这仅是有意义的，因为对整个模型进行了微调。
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        #仅当label不是None的时候才做此操作,否则，全部给-100，占位而已，因为预测会用到
        if example.label:
            if output_mode == "classification":
                # print(example.text_a, example.text_b, example.guid, example.label)
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label_id = -100

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("当label为-100时，仅是预测时，占位而已，没有用")

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cosmetics":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "newcos":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "components":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"mm-acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "cosmetics": CosmeticsProcessor,
    "newcos": NewcosProcessor,
    "components": NewComponentsProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "cosmetics": "classification",
    "newcos": "classification",
    "components": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "cosmetics": 3,
    "newcos": 3,
    "components": 2,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}
