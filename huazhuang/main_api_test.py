#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/23 5:02 下午
# @File  : main_api_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试main_api.py

import requests
import json

def dopredict(test_data, host="127.0.0.1"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}:5000/api/predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()

def dotrain(train_data, host="127.0.0.1"):
    """
    使用train_data训练模型
    :param train_data:
    :param host:
    :return:
    """
    url = f"http://{host}:5000/api/train"
    data = {'data': train_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()
if __name__ == '__main__':
    # test_data = [('补水效果：很好的补水效果，敷了绿茶很好用', '补水'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水')]
    # res = dopredict(test_data)
    # print(res)
    train_data = [('补水效果：很好的补水效果，敷了绿茶很好用', '补水', '积极'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化', '消极'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润', '中性'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感',  '中性'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水', '积极')]
    res = dotrain(train_data)
    print(res)