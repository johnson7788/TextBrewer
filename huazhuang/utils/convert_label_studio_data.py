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

def split_data(data, save_path, train_rate=0.9, test_rate=0.1):
    """
    保存为json文件，按比例保存
    train.json
    test.json
    :param data: data 列表,
    :return:
    """
    random.seed(30)
    random.shuffle(data)
    total = len(data)
    train_num = int(total * train_rate)
    train_file = os.path.join(save_path, 'train.json')
    test_file = os.path.join(save_path, 'test.json')
    train_data = data[:train_num]
    test_data = data[train_num:]
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    print(f"保存成功，训练集{len(train_data)},测试集{len(test_data)}")

if __name__ == '__main__':
    data = collect_json(dirpath="/opt/lavector")
    split_data(data=data, save_path="../data_root_dir/newcos")