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

def dopredict_macbert(host="127.0.0.1"):
    """
    预测结果
    :param test_data:
    :return:
    """
    testfile = "data_root_dir/newcos/dev.json"
    with open(testfile, 'r') as f:
        data = json.load(f)
    url = f"http://{host}:5000/api/predict_macbert"
    data = {'data': data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()


def dopredict_albert(host="127.0.0.1", data=None):
    """
    预测结果, components
    :param test_data:
    :return:
    """
    # testfile = "data_root_dir/newcos/dev.json"
    # with open(testfile, 'r') as f:
    #     data = json.load(f)
    url = f"http://{host}:5010/api/predict"
    data = {'data': data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()

if __name__ == '__main__':
    test_data = [('补水效果：很好的补水效果，敷了绿茶很好用', '绿茶'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水')]
    # res = dopredict(test_data)
    # print(res)
    # train_data = [['不想学习的一天来清理一下宿舍的空瓶图一这什么德国持妆粉底液是刚学化妆的时候买的 后面我皮肤越来越干 它也离我越来越远爱茉莉的唇釉有几只真的是又便宜又好看 不知道为什么每次网红色号反而没别的适合我。。图三这个唇釉颜色和它的外表一样可爱～是爱丽小屋吗太久远了记不得了科颜氏的高保湿水感觉还可以 艾诺碧天才水长痘时湿敷好像可以消炎 可惜了大瓶的放在家里都没带来红地球粉底液也是刚开始用感觉还行 后面也挺一般启初的牛奶谷胚面霜比另一个更贵的好用多了至本用着 感觉这个不知名唇膜有股奇怪的味道。。兰蔻粉水我是真的喜欢它的味道不过买了大瓶后总感觉有点刺激 准备出了自然共和国这卸妆水感觉卸不干净修丽可的物理防晒肤感好可怕。。Covermark粉霜也是用小样惊艳 买了就一般了科颜氏白泥是我用过最好涂匀的面膜！对我皮肤好像还是有点刺激 收起全文d', '保湿', '中性'], ['产品包装：非常精美，大气，上档次，高大上的亚子！适合肤质：一直使用，信赖百雀羚，持续用百雀羚产品已经七年了，非常好，很喜欢，nice！控油保湿：效果明显，一级棒！清洁效果：很干净，去油很棒！泡沫数量：起泡很多，用量少，实惠！其他特色：这款产品是我一直使用的，没有伤害症状！', '保湿', '积极'], ['这是第三购买了，碰上京东七夕活动也趁机买下，原来虽然还没有用完，因为活动就多囤点，面膜总是要用的反正，一直买的是这个牌子，用起来还是很放心的，没有酒精味，保湿效果很好，就是价钱小贵了些，如果平时再优惠点就更好了，不过这次没有送有点遗憾。', '没有酒精', '积极'], ['产品质感：打开一股特殊的味道，有点像酒精味，也有点像发酵的味道适合肤质：适合肤质：适合 敏感肌使用补水效果：补水效果不错。贴合效果：面膜大小正好，贴合面部非常好使用感受：总体来说还可以。本人敏感皮肤，用着不错', '质感', '中性'], ['用了一段时间，有没有效果不太清楚，就当去身体死皮了', '去身体死皮', '中性'], ['不想学习的一天来清理一下宿舍的空瓶图一这什么德国持妆粉底液是刚学化妆的时候买的 后面我皮肤越来越干 它也离我越来越远爱茉莉的唇釉有几只真的是又便宜又好看 不知道为什么每次网红色号反而没别的适合我。。图三这个唇釉颜色和它的外表一样可爱～是爱丽小屋吗太久远了记不得了科颜氏的高保湿水感觉还可以 艾诺碧天才水长痘时湿敷好像可以消炎 可惜了大瓶的放在家里都没带来红地球粉底液也是刚开始用感觉还行 后面也挺一般启初的牛奶谷胚面霜比另一个更贵的好用多了至本用着 感觉这个不知名唇膜有股奇怪的味道。。兰蔻粉水我是真的喜欢它的味道不过买了大瓶后总感觉有点刺激 准备出了自然共和国这卸妆水感觉卸不干净修丽可的物理防晒肤感好可怕。。Covermark粉霜也是用小样惊艳 买了就一般了科颜氏白泥是我用过最好涂匀的面膜！对我皮肤好像还是有点刺激 收起全文d', '白泥', '中性'], ['第一次使用，感觉还不错哟！有点缓解细纹的效果', '细纹', '积极'], ['套装设计很贴心，效果是不错的。芦荟镇定效果可以，刺鼻味是有的。操作容易-效果不错。缺点是漂色不到半月，颜色又开始悄咪咪的恢复了，估计2-3周要做一次。仅个人经验。', '芦荟', '中性'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '柔顺', '中性'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '茶花', '中性'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '氨基酸', '消极'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '顺滑', '消极'], ['孕期的时候就使用它家的海藻面膜，挺好用的', '海藻', '中性'], ['产品包装：包装精致，但是男人用的产品还应该有点男人的个性才好。适合肤质：很舒服，洗完脸感觉很润泽，洗完再使用面霜效果更好。控油保湿：洗完不油腻，能保持较长的时间。清洁效果：洗的很干净。泡沫数量：泡沫数量很丰富，用量比较省，我们这里的水质较硬，如果在南方水质软的地方可能会更好。', '控油', '中性'], ['泡沫数量：666666产品香味：麝香控油效果：#', '控油', '中性'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '硅', '消极'], ['产品包装：包装精致，但是男人用的产品还应该有点男人的个性才好。适合肤质：很舒服，洗完脸感觉很润泽，洗完再使用面霜效果更好。控油保湿：洗完不油腻，能保持较长的时间。清洁效果：洗的很干净。泡沫数量：泡沫数量很丰富，用量比较省，我们这里的水质较硬，如果在南方水质软的地方可能会更好。', '保湿', '消极'], ['【海藻面膜加牛奶可以淡斑】海藻面膜加牛奶调制后，可以增加皮肤的免疫力和自我保护力，皮肤上因为烈日暴晒而出现的晒斑，或者是因为作息混乱、睡眠不足而出现的色斑，都可以通过使用加牛奶的海藻面膜来淡化修复。 2安康?阴坡 ?', '淡斑', '中性'], ['想了想我最上头的香味还是檀香，上头 忍不住猛吸，大一的时候买了一套檀香味的洗发水，美吾发。味道真的好闻，每次洗完头被香味包围的感觉真好，味道非常舒服。结果过了几年，今年买小样的时候记成麝香，买回来一喷，嚯，一股男人味 ?', '麝香', '中性'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '修复受损头发', '中性'], ['第二瓶 提亮肤色还防晒 很方便 用着舒服 适合我这个大油皮', '提亮', '积极'], ['歪！这是什么宝藏洗护套装？我是个软件工程师，经常需要去客户公司演示软件，天气又毒又热，脸上出油出汗的，本来就油头的我头发也变得更油了，但发尾又很枯，真的很烦恼。还好最近入手了资生堂旗下丝蓓绮的沁润臻致修护套装，真的爱了爱了~光冲着好闻的花果香淡淡香槟金色外包装，就必须入手，洗发露和护发素摆在一起，包装正面看起来正好组成一个圆～嘻嘻～这个系列是金发膜衍生系列，洗发水是无硅油的，还含有双重氨基酸成分、山茶花精油，加上资生堂独特的革新渗透技术，可以有效修复受损头发，使发丝重回健康莹亮的状态。洗发水是透明质地，流动性很强，很水润，抹了护发素冲一遍，发丝真的柔顺了很多，两三天不洗头都不油啦，还geng有光泽了，有淡淡的花果香。头发也特别柔顺，和之前完全两种状态，简直就像去美发店做了护理一样。想和我一样做个干干净净香飘飘并且拥有清爽顺滑秀发的小仙女就赶紧入吧，在家就能享受到沙龙级别的护发体验～#护发##好物推荐##好物分享# 2杭州 收起全文d', '无硅油', '中性'], ['泡沫数量：666666产品香味：麝香控油效果：#', '麝香', '中性'], ['产品质感：打开一股特殊的味道，有点像酒精味，也有点像发酵的味道适合肤质：适合肤质：适合 敏感肌使用补水效果：补水效果不错。贴合效果：面膜大小正好，贴合面部非常好使用感受：总体来说还可以。本人敏感皮肤，用着不错', '补水', '消极'], ['近期，包括芦丹氏八月夜桂花，Burberry红粉恋歌，黛珂紫苏水牛油果哦，像剁手！ ?', '紫苏', '中性'], ['安放当下生活，追逐生活艺术它可以是书房把房间打扫得干干净净，泡上一杯清香的绿茶，坐在一张真皮座椅上，捧一本喜欢的书细细地品。书香夹着茶香使人清爽神怡，消去疲劳，一阵清风吹过，灵感在这里苏醒。短暂忘记生活工作里的忙，面对自己的心，倾听自己的声音，偷得浮生半日闲，在属于自己的域场里遨游一番。它可以是音乐间生活就像一首乐曲，每天都是一个乐音，把每个乐音串联起来才能谱写出优雅动听的旋律。关上房门，在这片属于自己的音乐空间里尽情舒展创造：一首吉他乐、一段架子鼓、一首优美的钢琴曲...打破紧张、单调的生活，追逐曾经的浪漫生活，精神也得到高度的放松和享受。?孔雀城创造美好生活它可以是运动场对坚持运动的人，岁月是偏爱的，它不仅放慢了步伐，还对他们温柔以待。喜欢丰富多样的健身活动，奈何只能疲惫穿梭于健身房和家之间，想解锁在家的肆意运动生活？你需要这样一间专属运动空间，在这里练瑜伽、跑步、深蹲、骑动感单车...健康的生活方式带给我们发自内心的满足与快乐。有朋自远方来，不亦乐乎朋友间的社交占据着生活中极大的比重，有时候，我们只需要一个安静的房间，与三两好友，谈论着时事热点，互相倾吐自己的心声。?孔雀城创造美好生活温暖的阳光布满整个书房，自由升降的榻榻米方桌给了大家舒适的置身空间，后面的书柜摆满各类书，聊累了亦可安静的沉浸在书海中，也不会显得尴尬。?空港新都孔雀城月鹭府102?湖景三居美宅南北通透户型方正三开间朝南沐浴更多阳光与清风超大瞰景阳台与南向飘窗尽览自然风光诠释生活的美好向往与追求~?孔雀城创造美好生活 收起全文d', '绿茶', '中性'], ['这是第三购买了，碰上京东七夕活动也趁机买下，原来虽然还没有用完，因为活动就多囤点，面膜总是要用的反正，一直买的是这个牌子，用起来还是很放心的，没有酒精味，保湿效果很好，就是价钱小贵了些，如果平时再优惠点就更好了，不过这次没有送有点遗憾。', '保湿', '中性'], ['第二瓶 提亮肤色还防晒 很方便 用着舒服 适合我这个大油皮', '防晒', '中性'], ['产品包装：非常精美，大气，上档次，高大上的亚子！适合肤质：一直使用，信赖百雀羚，持续用百雀羚产品已经七年了，非常好，很喜欢，nice！控油保湿：效果明显，一级棒！清洁效果：很干净，去油很棒！泡沫数量：起泡很多，用量少，实惠！其他特色：这款产品是我一直使用的，没有伤害症状！', '控油', '消极'], ['这是第三购买了，碰上京东七夕活动也趁机买下，原来虽然还没有用完，因为活动就多囤点，面膜总是要用的反正，一直买的是这个牌子，用起来还是很放心的，没有酒精味，保湿效果很好，就是价钱小贵了些，如果平时再优惠点就更好了，不过这次没有送有点遗憾。', '酒精', '中性'], ['不想学习的一天来清理一下宿舍的空瓶图一这什么德国持妆粉底液是刚学化妆的时候买的 后面我皮肤越来越干 它也离我越来越远爱茉莉的唇釉有几只真的是又便宜又好看 不知道为什么每次网红色号反而没别的适合我。。图三这个唇釉颜色和它的外表一样可爱～是爱丽小屋吗太久远了记不得了科颜氏的高保湿水感觉还可以 艾诺碧天才水长痘时湿敷好像可以消炎 可惜了大瓶的放在家里都没带来红地球粉底液也是刚开始用感觉还行 后面也挺一般启初的牛奶谷胚面霜比另一个更贵的好用多了至本用着 感觉这个不知名唇膜有股奇怪的味道。。兰蔻粉水我是真的喜欢它的味道不过买了大瓶后总感觉有点刺激 准备出了自然共和国这卸妆水感觉卸不干净修丽可的物理防晒肤感好可怕。。Covermark粉霜也是用小样惊艳 买了就一般了科颜氏白泥是我用过最好涂匀的面膜！对我皮肤好像还是有点刺激 收起全文d', '防晒', '中性'], ['近期，包括芦丹氏八月夜桂花，Burberry红粉恋歌，黛珂紫苏水牛油果哦，像剁手！ ?', '牛油果', '中性'], ['产品质感：打开一股特殊的味道，有点像酒精味，也有点像发酵的味道适合肤质：适合肤质：适合 敏感肌使用补水效果：补水效果不错。贴合效果：面膜大小正好，贴合面部非常好使用感受：总体来说还可以。本人敏感皮肤，用着不错', '酒精', '消极']]
    # res = dotrain(train_data)
    # print(res)
    # dopredict_macbert(host="192.168.50.119")
    # dopredict(host="127.0.0.1", test_data=test_data)
    # dopredict(host="127.0.0.1", test_data=test_data)
    # dopredict_albert(data=test_data)
    # dopredict_albert()