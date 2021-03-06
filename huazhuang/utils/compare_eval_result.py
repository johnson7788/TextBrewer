#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/31 4:01 下午
# @File  : compare_eval_result.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
import json
import os
import pandas as pd
import requests

def collect_data(devfile="../data_root_dir/newcos/dev.json", eval_results="../output_root_dir/newcos/eval_results-newcos.json"):
    """
    生成excel, 对比main.trainer.py生成的结果和devfile
    :param devfile: 训练文件，格式是 [(text, keyword, labels),..]
    :param eval_results:   main.trainer.py生成的文件output文件中的json文件   [(predid, probality)]
    :return:
    """
    labels = ["消极","中性","积极"]
    with open(devfile) as f:
        dev_data = json.load(f)
    with open(eval_results) as f:
        eval_data = json.load(f)
    assert len(dev_data) == len(eval_data)
    data = []
    for d, res in zip(dev_data, eval_data):
        one_data = {"text": d[0], "keyword":d[1], "label": d[2], "predict":labels[res[0]], "probability": format(res[1], "0.3f")}
        data.append(one_data)
    df = pd.DataFrame(data)
    excel_file = "result2.xlsx"
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    print(f"保存到excel成功{excel_file}")
    return data

def compare_model(hostname='http://127.0.0.1:3314'):
    """
    把收集到的数据，放到线上，对比一下准确率，不是和咱们自己的模型对比
    :param hostname:
    :return:
    """
    url = hostname + '/lavector/rest/aspect-sentiment-batch'
    headers = {'Content-Type': 'application/json'}
    mydata = collect_data()
    post_data = []
    for d in mydata:
        one = (d["text"], [d["keyword"]])
        post_data.append(one)
    data = {'channel': 'jd', 'data': post_data}
    print(f"发送请求到{url}, 数据量{len(post_data)}")
    res = requests.post(url, data=json.dumps(data), headers=headers)
    result = res.json()
    myresults = []
    for r in result['result']:
        keyword_list = list(r.keys())
        pres_list = list(r.values())
        assert  len(keyword_list) == 1
        assert  len(pres_list) == 1
        keyword = keyword_list[0]
        pres = pres_list[0]
        for k,v in pres.items():
            if v == 1:
                if k == "负向":
                    predict = "消极"
                elif k =="正向":
                    predict = "积极"
                else:
                    predict = "中性"
        myresults.append([keyword,predict])
    assert len(post_data) == len(myresults)

    #保存到文件
    newdata = []
    for d, res in zip(mydata, myresults):
        if res[0] != d["keyword"]:
            print(f"这条数据预测回来的关键字不一致{res[0]}")
            continue
        d["online_predict"] = res[1]
        newdata.append(d)
    df = pd.DataFrame(newdata)
    excel_file = "result_online.xlsx"
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    print(f"保存到excel成功{excel_file}")
    return newdata

def read_result_online():
    """
    读取result_online.xlsx,比较
    上文，关键字，下午的字数比
    pretext + keyword + posttest
    predict 表示的结果是75个字的，25+25+25的结果
    online_predict 表示的结果是 15+30+20
    :return:
    """
    df = pd.read_excel("result_online.xlsx")
    total = 0

    predict_yes = 0
    online_yes = 0
    for index, row in df.iterrows():
        label = row['label']
        predict = row['predict']
        online_predict = row['online_predict']
        if predict != online_predict:
            total += 1
            if predict == label:
                predict_yes +=1
            elif online_predict == label:
                online_yes +=1
            else:
                print("都没预测正确")
                print(row)
                print()
    print(f"共有{total}个不一样, 75个字预测的结果是{predict_yes}, 线上65个字的预测结果是{online_yes}")


def dopredict(test_data, url="http://127.0.0.1:5000/api/predict_macbert", proxy=True):
    """
    预测结果
    :param test_data:
    :return:
    """
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    if proxy:
        r = requests.post(url, data=json.dumps(data), headers=headers, timeout=360,
                           proxies=dict(http='socks5://127.0.0.1:9080', https='socks5://127.0.0.1:9080'))
    else:
        r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()

def download_data_and_compare(hostname=["http://192.168.50.139:8081/api/"], dirpath="/opt/lavector/absa/", jsonfile=["192.168.50.139_500_8081_0129.json"], isabsa=True, usecache=True):
    """
    从label_studio的某个hostname下载数据，然后预测，最后给出结果
    :return:
    """
    from absa_api import export_data
    #从label-studio下载文
    if usecache:
        json_files = [os.path.join(dirpath,j) for j in jsonfile]
    else:
        json_files = []
        for hname, jfile in zip(hostname,jsonfile):
            json_file = export_data(hostname=hname, dirpath=dirpath, jsonfile=jfile, proxy=False)
            json_files.append(json_file)
    original_data = []
    for json_file in json_files:
        #加载从label-studio获取的到json文件
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"共收集主机{json_file}的数据{len(data)} 条")
            original_data.extend(data)
    data = predict_comare_excel(original_data, isabsa=isabsa)
    return data


def download_data_and_compare_same(hostname=["http://192.168.50.139:8081/api/","http://192.168.50.139:8085/api/"], dirpath="/opt/lavector/absa/", jsonfile=["192.168.50.139_500_8081_0129.json","192.168.50.139_500_8085_0129.json"], isabsa=True):
    """
    对比相同的hostname的数据
    从label_studio的某个hostname下载数据，然后预测，最后给出结果
    :return:
    """
    from absa_api import export_data
    #从label-studio下载文
    if len(hostname) != 2:
        raise Exception("必须准2个hostname，里面包含相同的评估数据")
    result = []
    for hname, jfile in zip(hostname,jsonfile):
        original_data = []
        json_file = export_data(hostname=hname, dirpath=dirpath, jsonfile=jfile, proxy=False)
        #加载从label-studio获取的到json文件
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"共收集主机{hname}的数据{len(data)} 条")
            original_data.extend(data)
        predict_data, excel_data = predict_comare_excel(original_data, isabsa=isabsa)
        result.append([hname, predict_data, excel_data])
    #对比2个人标注的数据
    diffrent_data = []
    print(f"对比host为 {result[0][0], result[1][0]}")
    hname1, data1, pre1 = result[0]
    hname2, data2, pre2 = result[1]
    if len(data1) != len(data2):
        raise Exception("两个人标注的数据总数不一致")
    for d1, d2 in zip(data1,data2):
        if d1[0] != d2[0]:
            print("这条数据不一致")
        else:
            if d1[4] != d2[4]:
                print(f"2个人标注的标签不一致")
                print(d1[0])
                print(d1[1])
                print(d1[4])
                print(d2[4])
                one_data = {"text": d1[0], "keyword": d1[1], "P1_label": d1[4], "P2_label": d2[4], "location": d1[2:4]}
                diffrent_data.append(one_data)
    print(f"不一致的数据总量是{len(diffrent_data)}")
    df = pd.DataFrame(diffrent_data)
    writer = pd.ExcelWriter("diffrent.xlsx", engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    print(f"保存到diffrent.xlsx excel成功")
    return data

def predict_comare_excel(got_data,result_excel="result.xlsx", export_wrong_examples_excel="wrong.xlsx",correct_examples_excel= "correct.xlsx", isabsa=True):
    """
    :param got_data:
    :param result_excel:
    :param export_wrong_examples_excel:
    :param correct_examples_excel:
    :param isabsa:
    :return:  data是预处理后的，excel_data是模型预测的结果
    """
    from convert_label_studio_data import format_data, do_truncate_data
    # [(text, keyword, start_idx, end_idx, label)]
    original_data = format_data(got_data)
    # original_data, truncate_data, locations = do_truncate_data(data)
    if isabsa:
        predict_result = dopredict(test_data=original_data, url="http://192.168.50.139:5000/api/predict_truncate")
    else:
        predict_result = dopredict(test_data=original_data, url="http://192.168.50.139:5010/api/predict_truncate")
    # print(predict_result)
    excel_data = []
    for ori, d in zip(original_data, predict_result):
        one_data = {"text": ori[0], "keyword": ori[1], "label": ori[4], "predict": d[0], "location": d[3],
                    "probability": format(d[1], "0.3f"), "channel": ori[-2], "wordtype": ori[-1]}
        excel_data.append(one_data)
    df = pd.DataFrame(excel_data)
    writer = pd.ExcelWriter(result_excel, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    print(f"保存到excel成功{result_excel}")

    # 预测错误的样本
    predict_wrong_examples = []
    # 保存预测错误的样本到excel中
    correct_examples = []
    for ori, d in zip(original_data, predict_result):
        one_data = {"text": ori[0], "keyword": ori[1], "label": ori[4], "predict": d[0], "location": d[3],
                    "probability": format(d[1], "0.3f"), "channel": ori[-2], "wordtype": ori[-1]}
        if one_data["label"] != one_data["predict"]:
            print(f"{one_data['text']}: 模型预测的结果与ground truth不一致")
            predict_wrong_examples.append(one_data)
        else:
            correct_examples.append(one_data)
    print(f"总样本数是{len(original_data)},预测错误的样本总数是{len(predict_wrong_examples)}")
    print(f"总样本数是{len(original_data)},预测正确的样本总数是{len(correct_examples)}")

    df = pd.DataFrame(predict_wrong_examples)
    writer = pd.ExcelWriter(export_wrong_examples_excel, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='table1')
    writer.save()
    df = pd.DataFrame(correct_examples)
    writer = pd.ExcelWriter(correct_examples_excel, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='table1')
    writer.save()
    print(f"保存全部为错误的样本到excel: {export_wrong_examples_excel}完成")
    print(f"保存全部为正确的样本到excel: {correct_examples_excel}完成")
    print(f"准确率为{(len(correct_examples)) / len(original_data)}")
    return original_data, excel_data

def get_json_data_compare(jsonfile="/opt/lavector/192.168.50.119_8086.json"):
    """
    获取jsonfile，然后预测
    :return:
    """
    #加载从label-studio获取的到json文件
    data = predict_comare_excel(jsonfile, result_excel="result.xlsx", export_wrong_examples_excel="wrong.xlsx",correct_examples_excel= "correct.xlsx")
    return data

def all_bad_case():
    """
    测试所有的badcase
    总样本数是13688,预测错误的样本总数是1049
    总样本数是13688,预测正确的样本总数是12639
    保存全部为错误的样本到excel: wrong.xlsx完成
    保存全部为正确的样本到excel: correct.xlsx完成
    准确率为0.9233635300993571
    :return:
    """
    from convert_label_studio_data import collect_json
    data = collect_json(dirpath="/opt/lavector/absa")
    predict_comare_excel(got_data=data)

if __name__ == '__main__':
    # collect_data()
    # compare_model()
    # read_result_online()
    # download_data_and_compare()
    # download_data_and_compare(hostname=["http://192.168.50.139:8086/api/"], dirpath="/opt/lavector/components/", jsonfile= ["192.168.50.139_500_8086_0129.json"],isabsa=False)
    download_data_and_compare(hostname=["http://192.168.50.139:8081/api/","http://192.168.50.139:8085/api/"], dirpath="/opt/lavector/absa/", jsonfile= ["192.168.50.139_500_8081_0226.json","192.168.50.139_500_8085_0226.json"],isabsa=True)
    # get_json_data_compare()
    # download_data_and_compare_same()
    # all_bad_case()