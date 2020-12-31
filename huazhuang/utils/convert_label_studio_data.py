#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/30 4:49 下午
# @File  : convert_label_studio_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 把json文件转换成训练集和测试集，保存到data_root_dir目录下

import glob
import os
import json
import random
import re
import collections
import pandas as pd
import matplotlib.pyplot as plt


def collect_json(dirpath):
    """
    收集目录下的所有json文件，合成一个大的列表
    :param dirpath:
    :return: 返回列表
    """
    #所有文件的读取结果
    data = []
    search_file = os.path.join(dirpath, "*.json")
    # 搜索所有docx文件
    json_files = glob.glob(search_file)
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            print(f"{file}中包含数据{len(file_data)} 条")
            data.extend(file_data)
    print(f"共收集数据{len(data)} 条")
    return data


def truncate(input_text, max_len, trun_post='post'):
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

def do_truncate_data(data, left_max_seq_len=25, aspect_max_seq_len=25, right_max_seq_len=25):
    """
    对数据做truncate
    :param data:针对不同类型的数据进行不同的截断
    :return:返回列表，是截断后的文本，aspect
    所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
    和locations: [(start_idx,end_idx),...]
    """
    def aspect_truncate(content,aspect,aspect_start,aspect_end):
        """
        截断函数
        :param content:句子
        :param aspect:关键字
        :param aspect_start:开始位置
        :param aspect_end:结束位置
        :return:
        """
        text_left = content[:aspect_start]
        text_right = content[aspect_end:]
        text_left = truncate(text_left, left_max_seq_len)
        aspect = truncate(aspect, aspect_max_seq_len)
        text_right = truncate(text_right, right_max_seq_len, trun_post="pre")
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
    assert len(contents) == len(locations)
    print(f"截断的参数left_max_seq_len: {left_max_seq_len}, aspect_max_seq_len: {aspect_max_seq_len}, right_max_seq_len:{right_max_seq_len}。截断后的数据总量是{len(contents)}")
    return contents, locations

def split_data(data, save_path, train_rate=0.9, test_rate=0.05, dev_rate=0.05, weibodata=None):
    """
    保存为json文件，按比例保存
    train.json
    test.json
    dev.json
    :param data: data 列表,
    :param weibodata: 如果weibodata存在，只作为训练集使用
    :return:
    """
    random.seed(30)
    random.shuffle(data)
    total = len(data)
    train_num = int(total * train_rate)
    test_num = int(total * test_rate)
    train_file = os.path.join(save_path, 'train.json')
    dev_file = os.path.join(save_path, 'dev.json')
    test_file = os.path.join(save_path, 'test.json')
    train_data = data[:train_num]
    test_data = data[train_num:train_num+test_num]
    dev_data = data[train_num+test_num:]
    if weibodata:
        train_data = train_data.extend(weibodata)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f)
    print(f"保存成功，训练集{len(train_data)},测试集{len(test_data)},开发集{len(dev_data)}")

def split_data_dev(data, save_path, train_rate=0.8, dev_rate=0.2, weibodata=None):
    """
    保存为json文件，按比例保存
    train.json
    dev.json
    :param data: data 列表,
    :param weibodata: 如果weibodata存在，只作为训练集使用
    :return:
    """
    random.seed(30)
    random.shuffle(data)
    total = len(data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    train_file = os.path.join(save_path, 'train.json')
    dev_file = os.path.join(save_path, 'dev.json')
    train_data = data[:train_num]
    dev_data = data[train_num:]
    if weibodata:
        train_data.extend(weibodata)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f)
    print(f"保存成功，训练集{len(train_data)},开发集{len(dev_data)}")
    return train_data, dev_data


def format_data(data):
    """
    整理数据，只保留需要的字段
    :param data: [(text, keyword, start_idx, end_idx, label)]
    :return:
    """
    newdata = []
    for one in data:
        text = one['data']['text']
        keyword = one['data']['keyword']
        if not one['completions'][0]['result']:
            #过滤掉有问题的数据，有的数据是有问题的，所以没有标记，过滤掉
            continue
        start_idx = one['completions'][0]['result'][0]['value']['start']
        end_idx = one['completions'][0]['result'][0]['value']['end']
        label = one['completions'][0]['result'][0]['value']['labels'][0]
        new = (text,keyword,start_idx,end_idx,label)
        newdata.append(new)
    print(f"处理完成后的数据总数是{len(newdata)}")
    return newdata


def analysis_data(data):
    """
    分析数据
    :param data:
    :return:
    """
    texts, keywords, start_idxs, end_idxs, labels, lead_times = [], [], [], [], [], []
    for idx, one in enumerate(data):
        print(f"第{idx}条数据")
        text = one['data']['text']
        keyword = one['data']['keyword']
        if not one['completions'][0]['result']:
            #过滤掉有问题的数据，有的数据是有问题的，所以没有标记，过滤掉
            continue
        start_idx = one['completions'][0]['result'][0]['value']['start']
        end_idx = one['completions'][0]['result'][0]['value']['end']
        label = one['completions'][0]['result'][0]['value']['labels'][0]
        #标注的耗时
        lead_time = one['completions'][0]['lead_time']
        texts.append(text)
        keywords.append(keyword)
        start_idxs.append(start_idx)
        end_idxs.append(end_idx)
        labels.append(label)
        lead_times.append(lead_time)
    len_data = {"texts": list(map(len, texts)),
                "keywords": list(map(len, keywords)),
                "start_idxs": start_idxs,
                "end_idxs": end_idxs,
                "labels": list(map(len, labels)),
                "lead_times": lead_times,
                }
    df = pd.DataFrame(len_data)
    plt.plot(df)
    plt.ylabel('结果')
    plt.show()
    print(sorted(texts,key=len)[-10:])
    print(sorted(keywords,key=len)[-10:])
    print(sorted(lead_times)[:10])
    print(max(len_data["texts"]))

def checkis_alpha_digit(text):
    """
    如果text是英文或数字，返回True，否则返回False
    :param text:
    :return:
    """
    import re
    pattern = re.compile("[a-zA-Z0123456789]")
    if pattern.match(text):
        return True
    else:
        return False

def colect_weibo(save_file="../data_root_dir/cosmetics/all.txt", filter_english_keyword=False):
    """
    收集微博的训练数据
    :param save_file:
    :param filter_english_keyword: 是否过滤掉包含英文和数字字符的keyword
    :return: 存储到文件
    """
    # 原始文件中的sScore的映射方式
    class2id = {
        "消极": 0,
        "中性": 1,
        "积极": 2,
    }
    id2class = {value: key for key, value in class2id.items()}
    with open(save_file, 'r') as f:
        lines = f.readlines()
    data = []
    for idx, line in enumerate(lines):
        line_chinese = json.loads(line)
        text = line_chinese["content"]
        # 如果这个句子没有aspect，那就过滤掉
        if not line_chinese["aspect"]:
            continue
        for aspect in line_chinese["aspect"]:
            keyword = aspect["aspectTerm"]
            if filter_english_keyword:
                #过滤掉包含英文和数字的
                if checkis_alpha_digit(keyword):
                    #包含英文或数字
                    continue
            sScore = aspect["sScore"]
            label = id2class[sScore]
            if label not in list(id2class.values()):
                print(f"异常样本: {text}: {keyword}")
            start = aspect["start"]
            end = aspect["end"]
            # 验证一下单词的位置是否在newcontent中位置对应
            aspectTerm_insentence = "".join(text[start:end])
            if not keyword == aspectTerm_insentence:
                raise Exception(f"单词在句子中位置对应不上，请检查,句子行数{idx}, 句子是{line_chinese}")
            one = (text, keyword, start, end, label)
            data.append(one)
    print(f"总数据量是{len(data)}")
    return data

def valid_data(data):
    """
    验证数据
    :param data:
    :return:
    """
    labels = ['积极','消极','中性']
    length = len(data[0])
    for idx, d in enumerate(data):
        print(f"检查第{idx}条数据")
        if length == 5:
            content, aspect, aspect_start, aspect_end, label = d
        elif length == 3:
            content, aspect, label = d
        if label not in labels:
            print(f"数据的label不对: {label}")
        if len(label) == 0:
            print("数据的label长度为0")
        if len(content) == 0:
            print("数据的长度为0")
        if len(aspect) == 0:
            print("数据的aspect长度为0")
    print(f"数据校验成功")


def get_all_and_weibo_75():
    """
    25+25+25, 最高准确率79%
    最大75个字的长度的文本
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data)
    data = collect_json(dirpath="/opt/lavector")
    data = format_data(data)
    truncate_data, locations = do_truncate_data(data)
    train_data, dev_data = split_data_dev(data=truncate_data, save_path="../data_root_dir/newcos",weibodata=weibo_data_truncate)

def get_all_and_weibo_75_mini():
    """
    mini数据集, 共200条数据
    25+25+25, 最高准确率79%
    最大75个字的长度的文本
    :return:
    """
    data = collect_json(dirpath="/opt/lavector")
    data = format_data(data)
    truncate_data, locations = do_truncate_data(data)
    train_data, dev_data = split_data_dev(data=truncate_data[:200], save_path="../data_root_dir/newcos")


def get_all_and_weibo_45():
    """
    最大15+15+15个字的长度的文本,78%
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data,left_max_seq_len=15, aspect_max_seq_len=15, right_max_seq_len=15)
    data = collect_json(dirpath="/opt/lavector")
    data = format_data(data)
    truncate_data, locations = do_truncate_data(data,left_max_seq_len=15, aspect_max_seq_len=15, right_max_seq_len=15)
    train_data, dev_data = split_data_dev(data=truncate_data, save_path="../data_root_dir/newcos",weibodata=weibo_data_truncate)

def get_all_and_weibo_105():
    """
    40+40+25, 最高准确率
step: 1003 ****
 acc = 0.7807545106615636
step: 2006 ****
 acc = 0.7818480043739748
step: 3009 ****
 acc = 0.7807545106615636

    最大75个字的长度的文本
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data,left_max_seq_len=40, aspect_max_seq_len=25, right_max_seq_len=40)
    data = collect_json(dirpath="/opt/lavector")
    data = format_data(data)
    truncate_data, locations = do_truncate_data(data,left_max_seq_len=40, aspect_max_seq_len=25, right_max_seq_len=40)
    train_data, dev_data = split_data_dev(data=truncate_data, save_path="../data_root_dir/newcos",weibodata=weibo_data_truncate)


if __name__ == '__main__':
    # get_all_and_weibo_105()
    get_all_and_weibo_75_mini()