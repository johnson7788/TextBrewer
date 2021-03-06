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
import requests


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

def saveto_excel():
    """
    保存collect_json的数据到excel
    :return:
    """
    output_file = "export.xlsx"
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    #得到的数据 (text, keyword, start_idx, end_idx, label,channel,wordtype)
    data,cancel_data = get_all(absa=False, keep_cancel=True, split=False)
    df = pd.DataFrame(data,columns=['Text','Keyword', 'Start', 'End', 'Label', 'channel', 'wordtype'])
    df.to_excel(writer, sheet_name='标注数据')
    df1 = pd.DataFrame(cancel_data,columns=['Text','Keyword', 'Start', 'End', 'Label', 'channel', 'wordtype'])
    df1.to_excel(writer, sheet_name='取消数据')
    writer.save()
    print(f"保存到excel: {output_file}完成")


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
    #保留原始数据，一同返回
    original_data = []
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
                original_data.append(one_data)
        elif len(one_data) == 3:
            #不带aspect关键字的位置信息，带label
            content, aspect, label = one_data
            iter = re.finditer(aspect, content)
            for m in iter:
                aspect_start, aspect_end = m.span()
                new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
                contents.append((new_content, aspect, label))
                locations.append((aspect_start,aspect_end))
                original_data.append(one_data)
        elif len(one_data) == 4:
            # 不带label时，长度是4，
            content, aspect, aspect_start, aspect_end = one_data
            new_content = aspect_truncate(content, aspect, aspect_start,aspect_end)
            contents.append((new_content, aspect))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 5:
            content, aspect, aspect_start, aspect_end, label = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
            contents.append((new_content, aspect, label))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 7:
            content, aspect, aspect_start, aspect_end, label, channel,wordtype = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
            contents.append((new_content, aspect, label))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        else:
            raise Exception(f"这条数据异常: {one_data},数据长度或者为2, 4，或者为5")
    assert len(contents) == len(locations) == len(original_data)
    print(f"截断的参数left_max_seq_len: {left_max_seq_len}, aspect_max_seq_len: {aspect_max_seq_len}, right_max_seq_len:{right_max_seq_len}。截断后的数据总量是{len(contents)}")
    return original_data, contents, locations

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


def format_data(data, keep_cancel=False):
    """
    整理数据，只保留需要的字段
    :param data:,
    :param keep_cancel: 是否返回cancel的数据，如果是True，那么返回data, cancel_data，否则返回data
    :return:[(text, keyword, start_idx, end_idx, label,channel,wordtype)]
    """
    newdata = []
    #未标注，取消标注的数据
    cancel_data = []
    #多人标注的条数, 一共多人标注的条数
    repeats_completions = 0
    #多人标注时，不一致的数量
    not_same_label_num = 0
    for one in data:
        text = one['data']['text']
        keyword = one['data']['keyword']
        channel = one['data']['channel']
        wordtype= one['data']['wordtype']
        if not one['completions'][0]['result']:
            #过滤掉有问题的数据，有的数据是有问题的，所以没有标记，过滤掉
            continue
        # 校验一下一共几个关键字，标注了几个
        research = re.findall(keyword,text)
        if len(research) > 1:
            annotation_num = len(one['completions'][0]['result'])
            if len(research) != annotation_num:
                print("标注的数量不匹配, 不会造成影响，例如句子中有2个词，我们只标注了一个词")
        #一句话中的多个标注结果
        #如果有多个标注结果，对比下标注结果，把不同的打印出来,
        canceled = []
        # if len(one['completions']) >1:
        # print(f"标注结果大于1个，打印出来，只打印出2个人标注完全不一样的结果")
        unique = {}
        #存储不被取消和没有重复的标注的数据
        results = []
        #多人标注数据统计
        if len(one['completions']) >1:
            repeats_completions +=1
        for completion in one['completions']:
            if completion.get("was_cancelled"):
                canceled.extend(completion['result'])
                continue
            for res in completion['result']:
                start_idx = res['value']['start']
                end_idx = res['value']['end']
                label = res['value']['labels'][0]
                unique_sentence_idx = f"{str(start_idx):{str(end_idx)}}"
                unique_sentence_label= f"{label}:{keyword}"
                if unique_sentence_idx in unique.keys():
                    if unique_sentence_label != unique[unique_sentence_idx]:
                        print(f"2个人标注的数据不一致, 句子为:{text}   标注的单词位置: {start_idx}-{end_idx}   一个标注为:{unique[unique_sentence_idx]}  另一个标注为:{unique_sentence_label},标注不一致的被丢弃")
                        not_same_label_num += 1
                    else:
                        # 2个人标注的一致,忽略
                        print(f"2个人标注的数据一致,标注了多次,句子为:{text}   标注的单词位置: {start_idx}-{end_idx}   标注为:{unique[unique_sentence_idx]}")
                        continue
                else:
                    unique[unique_sentence_idx] = unique_sentence_label
                    results.append(res)
        if canceled:
            #如果canceled有数据，那么整个数据都不要了，直接cancel, 那么我们保留一个cancel数据就可以了
            res = canceled[0]
            start_idx = res['value']['start']
            end_idx = res['value']['end']
            label = "CANCELLED"
            new = (text, keyword, start_idx, end_idx, label, channel, wordtype)
            cancel_data.append(new)
        else:
            for res in results:
                start_idx = res['value']['start']
                end_idx = res['value']['end']
                label = res['value']['labels'][0]
                new = (text,keyword,start_idx,end_idx,label,channel,wordtype)
                newdata.append(new)
    print(f"处理完成后的数据总数是{len(newdata)}, 存在{repeats_completions}条多人标注的数据, 标注不一致的数据有{not_same_label_num}条, 被跳过的数据有{len(cancel_data)}")
    if keep_cancel:
        return newdata, cancel_data
    else:
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
    :return: 存储到文件, [(text, keyword, start_idx, end_idx, label, channel, type),...]
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
            one = (text, keyword, start, end, label, 'weibo', '品牌')
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

def dopredict_macbert(data, host="127.0.0.1"):
    """
    预测结果
    :param data:
    :return: [(predicted_label, predict_score),...]
    """
    url = f"http://{host}:5000/api/predict_macbert"
    data = {'data': data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()

def model_filter_again():
    """
    用模型筛选出所有预测错误的样本，重新检查
    只获取已标注的数据
    25+25+25, 最高准确率79%
    最大75个字的长度的文本
    :return: [['Text','Keyword', 'Start', 'End','Label','Predict', 'Score'],...]
    """
    data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(data)
    original_data, truncate_data, locations = do_truncate_data(data)
    response_data = dopredict_macbert(data=truncate_data, host="192.168.50.119")
    #预测错误的样本
    predict_wrong_examples = []
    # 保存样本到csv中
    predict_wrong_examples_csv = "wrong.json"
    # 保存预测错误的样本到excel中
    export_wrong_examples_excel = "wrong.xlsx"
    for d,t,r in zip(data, truncate_data, response_data):
        text, keyword, start_idx, end_idx, label = d
        new_content, aspect, _ = t
        predicted_label, predict_score, _ = r
        if aspect != keyword:
            print("请检查, truncate之后数据混乱了")
        if label != predicted_label:
            print("模型预测的结果与ground truth不一致")
            predict_wrong_examples.append([text, keyword, start_idx, end_idx, label, predicted_label, predict_score])
    print(f"总样本数是{len(data)},预测错误的样本总数是{len(predict_wrong_examples)}")

    with open(predict_wrong_examples_csv, 'w') as f:
        json.dump(predict_wrong_examples, f)
    df = pd.DataFrame(predict_wrong_examples,columns=['Text','Keyword', 'Start', 'End','Label','Predict', 'Score'])
    writer = pd.ExcelWriter(export_wrong_examples_excel, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='table1')
    writer.save()
    print(f"保存到excel: {export_wrong_examples_excel}完成")

def get_all_and_weibo_75():
    """
    25+25+25, 最高准确率79%
    最大75个字的长度的文本
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    original_data, weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data)
    data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(data)
    original_data, truncate_data, locations = do_truncate_data(data)
    train_data, dev_data = split_data_dev(data=truncate_data, save_path="../data_root_dir/newcos",weibodata=weibo_data_truncate)

def get_all_and_weibo():
    """
    不要做truncate，仅保留原文本, 保存到新的位置cosmetrics, 使用不太相同的训练方法
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    data = collect_json(dirpath="/opt/lavector/absa")
    studio_data = format_data(data)
    train_data, dev_data = split_data_dev(data=studio_data, save_path="../data_root_dir/cosmetics",weibodata=weibo_data)

def get_all(absa=True, keep_cancel=False, split=True):
    """
    不要做truncate，仅保留原文本
    :return:
    """
    if absa:
        dirpath = "/opt/lavector/absa"
    else:
        dirpath = "/opt/lavector/components"
    data = collect_json(dirpath)
    studio_data = format_data(data,keep_cancel)
    if split:
        if absa:
            train_data, dev_data = split_data_dev(data=studio_data, save_path="../data_root_dir/cosmetics")
        else:
            train_data, dev_data = split_data_dev(data=studio_data, save_path="../data_root_dir/components")
    return studio_data

def get_all_and_weibo_75_mini():
    """
    mini数据集, 共200条数据
    25+25+25, 最高准确率79%
    最大75个字的长度的文本
    :return:
    """
    data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(data)
    original_data, truncate_data, locations = do_truncate_data(data)
    train_data, dev_data = split_data_dev(data=truncate_data[:200], save_path="../data_root_dir/newcos")


def get_all_and_weibo_45():
    """
    最大15+15+15个字的长度的文本,78%
    :return:
    """
    weibo_data = colect_weibo(filter_english_keyword=True)
    original_data, weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data,left_max_seq_len=15, aspect_max_seq_len=15, right_max_seq_len=15)
    data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(data)
    original_data, truncate_data, locations = do_truncate_data(data,left_max_seq_len=15, aspect_max_seq_len=15, right_max_seq_len=15)
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
    original_data, weibo_data_truncate, weibo_locations = do_truncate_data(weibo_data,left_max_seq_len=40, aspect_max_seq_len=25, right_max_seq_len=40)
    data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(data)
    original_data ,truncate_data, locations = do_truncate_data(data,left_max_seq_len=40, aspect_max_seq_len=25, right_max_seq_len=40)
    train_data, dev_data = split_data_dev(data=truncate_data, save_path="../data_root_dir/newcos",weibodata=weibo_data_truncate)


def components_data_hive():
    """
    从hive获取数据
    :return:[{'channel': 'jd', 'text': '还没用，烟酰胺原液还没用完，用着还不错，吸收很快，没有那种粘粘的感觉。还是朋友介绍的这个牌子，她说效果很好，买了好几套了，爽肤水和乳液用着也很好。', 'keyword': '烟酰胺', 'wordtype': '成分'},
            ...
            ]
    """
    from read_hive import get_absa_corpus
    data = get_absa_corpus(channel=None,requiretags=['component'], number=1000, not_cache=False, save_cache=False,table="da_wide_table_new")
    return data


def make_static(data):
    """
    统计正负样本的比例
    :param data: [[sentence, label],...]
    :return:
    """
    from collections import Counter
    keywords = [d[1] for d in data]
    labels = [d[2] for d in data]
    lcoutner = Counter(labels)
    kcounter = Counter(keywords)
    print(f"标签统计: {lcoutner}")
    print(f"关键字统计: {kcounter}")

def get_components_75():
    """
    25+25+25,
    最大75个字的长度的文本
    :return:
    """
    #或者从文件夹中获取json数据
    data = collect_json(dirpath="/opt/lavector/components")
    data = format_data(data)
    original_data, truncate_data, locations = do_truncate_data(data)
    make_static(truncate_data)
    train_data, dev_data = split_data_dev(data=truncate_data, train_rate=0.9, dev_rate=0.1, save_path="../data_root_dir/components/")

def gen_components_75_first():
    """
    第一次生成数据,随机label,
    :return:
    """
    #从hive获取数据
    data = components_data_hive()
    format_data = [[d['text'], d['keyword'],random.choice(["是","否"])] for d in data]
    original_data, truncate_data, locations = do_truncate_data(format_data)
    split_data(data=truncate_data, save_path="../data_root_dir/components/")

def db2local(save_file="/opt/lavector/components/json.mongo", collect='label_corpus', use_cache=True):
    """
    从MongoDB 获取数据，保存到save_file中, mongo结果， 模型准确率为:0.6782511210762332
    :param save_file:
    :param collect: label_corpus 获取成分的词的标注
    :return:
    """
    # 配置client, 如果存在，并且使用cache，那么直接返回结果
    if use_cache and os.path.exists(save_file):
        with open(save_file, 'r') as f:
            results = json.load(f)
        return results
    results = []
    import pymongo
    client = pymongo.MongoClient("192.168.50.139", 27017)
    # 设置database
    db = client['ai-corpus']
    # 选择哪个collections
    collection = db[collect]
    mydoc = collection.find({})
    for x in mydoc:
        content = x["content"]
        for aspect in x["aspect"]:
            start = aspect["start"]
            end = aspect["end"]
            keyword = aspect["aspectTerm"]
            aspectCategory = aspect["aspectCategory"]
            if aspectCategory == "其他":
                label = "否"
            else:
                label = "是"
            results.append([content, keyword, start, end, label])
    with open(save_file, 'w') as f:
        json.dump(results,f)
    print(f"获得样本数{len(results)}，文件已生成{save_file}")
    return results

def dopredict_albert(host="127.0.0.1", data=None):
    """
    预测成分的结果
    :param test_data:
    :return:
    """
    # testfile = "data_root_dir/newcos/dev.json"
    # with open(testfile, 'r') as f:
    #     data = json.load(f)
    url = f"http://{host}:5010/api/predict_albert"
    data = {'data': data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()

def mongo_res_to_excel(mongo_res="mongo.res"):
    """
    把mongo中的获取的数据预测后的结果，统计并生成excel
    :param mongo_res:
    :return:
    """
    pass

def save2mongo_absa():
    """
    保存数据到mongo
    :param data:
    :return:
    """
    import pymongo
    studio_data = collect_json(dirpath="/opt/lavector/absa")
    data = format_data(studio_data)
    client = pymongo.MongoClient("192.168.50.139", 27017)
    # 设置database
    db = client['ai-corpus']
    # 选择哪个collections
    collection = db['as_corpus']
    class2id = {
        "消极": 0,
        "中性": 1,
        "积极": 2,
    }
    before_num = collection.count()
    print(f"数据库中已有数据{before_num}条, 待插入数据{len(data)}条")
    mdata = []
    for one in data:
        text, keyword, start_idx, end_idx, label, channel, wordtype = one
        label_id = class2id[label]
        one_data = {
        "fileName" : "unknown",
        "content" : text,
        "start" : 0,
        "end" : len(text),
        "aspect" : [
                {
                    "aspectTermId" : "T1",
                    "aspectTerm" : wordtype,
                    "aspectCategory" : "单个对象",
                    "start" : start_idx,
                    "end" : end_idx,
                    "sScore" : label_id,
                    "contentAspectTerm" : keyword
                }
            ],
        "source" : channel,
        "industry" : "美妆",
        "brand" : "unknown",
        "batchId" : "20210224",
        "createTime": "2021-02-24 10:47:48",
        "updateTime": "2021-02-24 10:47:48"
    }
        mdata.append(one_data)
    print(f"待插入数据整理后{len(mdata)}条")
    collection.insert_many(mdata)
    after_num = collection.count()
    print(f"插入后数据有{after_num},共插入数据{after_num-before_num}")

def save2mongo_components():
    """
    保存成分数据到mongo
    :param data:
    :return:
    """
    import pymongo
    studio_data = collect_json(dirpath="/opt/lavector/components")
    data = format_data(studio_data)
    client = pymongo.MongoClient("192.168.50.139", 27017)
    # 设置database
    db = client['ai-corpus']
    # 选择哪个collections
    collection = db['label_corpus']
    class2id = {
        "是": "T",
        "否": "F",
    }
    before_num = collection.count()
    print(f"数据库中已有数据{before_num}条, 待插入数据{len(data)}条")
    mdata = []
    for one in data:
        text, keyword, start_idx, end_idx, label, channel, wordtype = one
        label_id = class2id[label]
        one_data = {
        "content" : text,
        "start" : 0,
        "end" : len(text),
        "aspect" : [
            {
                "aspectTermId" : "T1",
                "id" : 1,
                "aspectTerm" : keyword,
                "aspectCategory" : wordtype,
                "start" : start_idx,
                "end" : end_idx,
                "bratLabel" : label_id
            }
        ],
        "fileName" : "",
        "source" : channel,
        "industry" : "mz",
        "brand" : "",
        "batchId" : "20210225",
        "createTime" : "2021-02-25 09:32:37",
        "updateTime" : "2021-02-25 09:32:37"
    }
        mdata.append(one_data)
    print(f"待插入数据整理后{len(mdata)}条")
    collection.insert_many(mdata)
    after_num = collection.count()
    print(f"插入后数据有{after_num},共插入数据{after_num-before_num}")


if __name__ == '__main__':
    # get_all_and_weibo_105()
    # get_all_and_weibo_75_mini()
    # get_all_and_weibo_75()
    # saveto_excel()
    # model_filter_again()
    # data = collect_json(dirpath="/opt/lavector/absa")
    # data = format_data(data)
    # get_components_75()
    # data = db2local(use_cache=False)
    # dopredict_albert(host="127.0.0.1", data=data)
    # result = dopredict_albert(host="192.168.50.139", data=data)
    get_all_and_weibo()
    # get_all(absa=True, keep_cancel=False,split=True)
    # get_all(absa=False, keep_cancel=True)
    # save2mongo()
    # saveto_excel()
    # save2mongo_components()
    # get_all(absa=False)