#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/31 4:01 下午
# @File  : compare_eval_result.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
import json
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


def dopredict(test_data, host="127.0.0.1"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}:5000/api/predict_macbert"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()

def download_data_and_compare(hostname="http://192.168.50.119:8080/api/", dirpath="/opt/lavector/", jsonfile="192.168.50.119_500.json"):
    """
    从label_studio的某个hostname下载数据，然后预测，最后给出结果
    :return:
    """
    from absa_api import export_data
    #从label-studio下载文
    json_file = export_data(hostname=hostname,dirpath=dirpath, jsonfile=jsonfile, proxy=False)
    data = predict_comare_excel(json_file)
    return data
def predict_comare_excel(json_file,result_excel="result.xlsx", export_wrong_examples_excel="wrong.xlsx",correct_examples_excel= "correct.xlsx"):
    from convert_label_studio_data import format_data, do_truncate_data
    #加载从label-studio获取的到json文件
    with open(json_file, 'r') as f:
        data = json.load(f)
        print(f"共收集数据{len(data)} 条")
    # [(text, keyword, start_idx, end_idx, label)]
    data = format_data(data)
    original_data,truncate_data, locations = do_truncate_data(data)
    predict_result = dopredict(test_data=truncate_data, host="192.168.50.119")
    # print(predict_result)
    excel_data = []
    for ori, d, loc in zip(original_data, predict_result, locations):
        one_data = {"text": ori[0], "keyword": d[2][1], "label": d[2][2], "predict": d[0], "location": loc,
                    "probability": format(d[1], "0.3f"), "channel":ori[-2], "wordtype":ori[-1]}
        excel_data.append(one_data)
    df = pd.DataFrame(excel_data)
    writer = pd.ExcelWriter(result_excel, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    print(f"保存到excel成功{result_excel}")

    #预测错误的样本
    predict_wrong_examples = []
    # 保存预测错误的样本到excel中
    correct_examples = []
    for ori, d, loc in zip(original_data, predict_result, locations):
        one_data = {"text": ori[0], "keyword": d[2][1], "label": d[2][2], "predict": d[0], "location": loc,
                    "probability": format(d[1], "0.3f"), "channel":ori[-2], "wordtype":ori[-1]}
        if one_data["label"] != one_data["predict"]:
            print(f"{one_data['text']}: 模型预测的结果与ground truth不一致")
            predict_wrong_examples.append(one_data)
        else:
            correct_examples.append(one_data)
    print(f"总样本数是{len(data)},预测错误的样本总数是{len(predict_wrong_examples)}")
    print(f"总样本数是{len(data)},预测正确的样本总数是{len(correct_examples)}")

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
    print(f"准确率为{(len(correct_examples))/len(data)}")
    return data


def get_json_data_compare(jsonfile="/opt/lavector/192.168.50.119_8086.json"):
    """
    获取jsonfile，然后预测
    :return:
    """
    #加载从label-studio获取的到json文件
    data = predict_comare_excel(jsonfile, result_excel="result.xlsx", export_wrong_examples_excel="wrong.xlsx",correct_examples_excel= "correct.xlsx")
    return data



if __name__ == '__main__':
    # collect_data()
    # compare_model()
    # read_result_online()
    download_data_and_compare()
    # get_json_data_compare()