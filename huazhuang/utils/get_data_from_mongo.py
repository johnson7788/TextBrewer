import json
import random
import os


def db2local(save_file):
    """
    从MongoDB 获取数据，保存到save_file中
    :param save_file:
    :return:
    """
    # 配置client
    import pymongo
    client = pymongo.MongoClient("192.168.50.139", 27017)
    # 设置database
    db = client['ai-corpus']
    # 选择哪个collections
    collection = db['as_corpus']
    mydoc = collection.find({})
    with open(save_file, 'w') as f:
        for x in mydoc:
            x.pop('_id')
            content = json.dumps(x)
            f.write(content + '\n')
    print(f"文件已生成{save_file}")

def split_all(save_file, train_rate=0.9, test_rate=0.1):
    """
    拆分成90%训练集，10%测试集
    :param save_file:
    :param train_rate: float
    :param test_rate:
    :return:
    """
    random.seed(30)
    examples = []
    with open(save_file, 'r') as f:
        lines = f.readlines()
        # 每3行一个样本
        for i in range(0, len(lines), 3):
            examples.append((lines[i], lines[i + 1], lines[i + 2]))
    random.shuffle(examples)
    total = len(examples)
    train_num = int(total * train_rate)
    test_num = int(total * test_rate)
    train_file = os.path.join(os.path.dirname(save_file), 'train.txt')
    test_file = os.path.join(os.path.dirname(save_file), 'test.txt')
    with open(train_file, 'w') as f:
        for x in examples[:train_num]:
            f.write(x[0])
            f.write(x[1])
            f.write(x[2])
    with open(test_file, 'w') as f:
        for x in examples[train_num:]:
            f.write(x[0])
            f.write(x[1])
            f.write(x[2])
    print(f"文件已生成\n {train_file}, 样本数: {train_num} \n {test_file}, 样本数: {test_num}")

def textsentiment_process(save_file, new_file, truncate=None):
    """
    类似
    $T$ is super fast , around anywhere from 35 seconds to 1 minute .
    Boot time
    1
    :param save_file:
    :param new_file: 存储到新文件
    :param truncate: 截断处理，截断前后，默认为None，可以为int，截断保留数字
    :return: 存储到文件
    """
    # 原始文件中的sScore的映射方式
    class2id = {
        "NEG": 0,
        "NEU": 1,
        "POS": 2,
    }
    id2class = {value: key for key, value in class2id.items()}
    with open(save_file, 'r') as f:
        lines = f.readlines()
    # 打印多少条样本
    print_example = 10
    # 总数据量
    total = 0
    with open(new_file, 'w') as f:
        for line in lines:
            line_chinese = json.loads(line)
            # 使用 $T$代表apsect
            content = line_chinese["content"]
            # 如果这个句子没有aspect，那就过滤掉
            if not line_chinese["aspect"]:
                continue
            for aspect in line_chinese["aspect"]:
                aspectTerm = aspect["aspectTerm"]
                sScore = aspect["sScore"]
                start = aspect["start"]
                end = aspect["end"]
                # 验证一下单词的位置是否在newcontent中位置对应
                aspectTerm_insentence = "".join(content[start:end])
                if not aspectTerm == aspectTerm_insentence:
                    raise Exception(f"单词在句子中位置对应不上，请检查,句子行数{total}, 句子是{line_chinese}")
                if truncate:
                    #如果truncate为数字，那么开始截断
                    if truncate > start:
                        left = content[:start]
                    else:
                        left = content[start-truncate:start]
                    right = content[end:end+truncate]
                    line1 = left + "$T$" + right
                else:
                    line1 = content[:start] + "$T$" + content[end:]
                line2 = aspectTerm
                # sScore映射成我们需要的, -1，0，1格式
                line3 = str(sScore - 1)
                if print_example > 0:
                    print(line1)
                    print(line2)
                    print(line3)
                    print_example -= 1
                total += 1
                f.write(line1 + "\n")
                f.write(line2 + "\n")
                f.write(line3 + "\n")
    print(f"文件已生成{new_file}, 总数据量是{total}")

def do_truncate_special(content, keyword, start_idx, end_idx, add_special=True, max_seq_length=70, verbose=True,special= '_'):
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
        print(len(content),content)
        print("收到的样本关键字和位置信息")
        print(keyword, start_idx, end_idx)
    #因为我们的结构是 CLS_texta_+SEP+keyword+SEP
    max_texta_length = max_seq_length- len(keyword) - 3
    # 我们尝试在原句子中MASK掉关键字,但如果句子长度大于我们模型中的最大长度，我们需要截断
    if content[start_idx:end_idx] != keyword:
        print(f"关键字的位置信息不准确: {content}: {keyword}:  {start_idx}:{end_idx}, 获取的关键字是 {content[start_idx:end_idx] }")
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
        right_text = texta_special[end_idx + len(special)*2:]
    else:
        right_text = texta_special[end_idx:]
    # 开始检查长度, 如果长度大于最大序列长度，我们要截取关键字上下的句子，直到满足最大长度以内,截取时用句子分隔的方式截取
    if len(texta_special) > max_texta_length:
        # 需要对texta进行阶段,采取怎样的截断方式更合适,按逗号和句号和冒号分隔
        texta_split = re.split('([：。；，])', texta_special)
        # 确定keyword在列表中的第几个元素中
        special_keyword_idx_list = []
        for t_idx, t in enumerate(texta_split):
            if special_keyword in t:
                special_keyword_idx_list.append(t_idx)
        key_symbol = False
        #和start_idx比较，确定是哪个keyword在列表哪个元素中
        if len(special_keyword_idx_list) >1:
            for kidx in special_keyword_idx_list:
                before_len = sum(len(i) for i in texta_split[:kidx])
                all_idx = [m.start() for m in re.finditer(special_keyword, texta_split[kidx])]
                for before_spe_len in all_idx:
                    if before_len + before_spe_len == start_idx:
                        special_keyword_idx = kidx
                        break
        elif len(special_keyword_idx_list) ==1 :
            special_keyword_idx = special_keyword_idx_list[0]
        else:
            #没有找到关键字，切分之后, 如果关键字中有，。：,那么也是有问题的，只好强制截断
            key_symbol =True

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
    if (len(left_text)+ len(special_keyword) + len(right_text)) >max_seq_length:
        raise Exception("处理后总长度序列长度太长")
    if len(text_a) >max_seq_length:
        raise Exception("处理后text_a序列长度太长")
    if not add_special and text_a not in content:
        raise Exception("处理完成后文本不在content中了")
    return text_a, special_keyword, left_text, right_text


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

def aspect_truncate(content, aspect,aspect_start,aspect_end, left_max_seq_len=25,aspect_max_seq_len=25,right_max_seq_len=25):
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
    text_left = truncate(text_left, left_max_seq_len)
    aspect = truncate(aspect, aspect_max_seq_len)
    text_right = truncate(text_right, right_max_seq_len, trun_post="pre")
    new_content = text_left + aspect + text_right
    return new_content, text_left,text_right

def components_process(train_file, test_file, max_seq_length=70):
    """
    类似
    $T$ is super fast , around anywhere from 35 seconds to 1 minute .
    Boot time
    1
    :param train_file: 0.8的比例存储，
    :param test_file: 0.2
    :param truncate: 截断处理，截断前后，默认为None，可以为int，截断保留数字
    :return: 存储到文件
    """
    from convert_label_studio_data import get_all
    data = get_all(absa=False, keep_cancel=False, split=False)
    # 原始文件中的sScore的映射方式
    class2id = {
        "否": 0,
        "是": 1,
    }
    id2class = {value: key for key, value in class2id.items()}
    train_num = int(len(data) *0.8)
    train_data = data[:train_num]
    dev_data = data[train_num:]
    def save2file(filename, data):
        # 打印多少条样本
        print_example = 10
        # 总数据量
        total = 0
        with open(filename, 'w') as f:
            for line in data:
                text, keyword, start_idx, end_idx, label, channel, wordtype = line
                # 使用 $T$代表apsect
                text_a, left_text,right_text = aspect_truncate(text,keyword,start_idx,end_idx)
                # sScore映射成我们需要的, -1，0，1格式
                text = left_text + "$T$" + right_text
                labelid = str(class2id[label])
                if print_example > 0:
                    print(text)
                    print(keyword)
                    print(labelid)
                    print_example -= 1
                total += 1
                f.write(text + "\n")
                f.write(keyword + "\n")
                f.write(labelid + "\n")
        print(f"文件已生成{filename}, 总数据量是{total}")
    save2file(filename=train_file, data=train_data)
    save2file(filename=test_file, data=dev_data)

def components_process_truancate(train_file, test_file, max_seq_length=70):
    """
    类似
    $T$ is super fast , around anywhere from 35 seconds to 1 minute .
    Boot time
    1
    :param train_file: 0.8的比例存储，
    :param test_file: 0.2
    :param truncate: 截断处理，截断前后，默认为None，可以为int，截断保留数字
    :return: 存储到文件
    """
    from convert_label_studio_data import get_all
    data = get_all(absa=False, keep_cancel=False, split=False)
    # 原始文件中的sScore的映射方式
    class2id = {
        "否": 0,
        "是": 1,
    }
    id2class = {value: key for key, value in class2id.items()}
    train_num = int(len(data) *0.8)
    train_data = data[:train_num]
    dev_data = data[train_num:]
    def save2file(filename, data):
        # 打印多少条样本
        print_example = 10
        # 总数据量
        total = 0
        with open(filename, 'w') as f:
            for line in data:
                text, keyword, start_idx, end_idx, label, channel, wordtype = line
                # 使用 $T$代表apsect
                text_a, special_keyword, left_text,right_text = do_truncate_special(text,keyword,start_idx,end_idx,add_special=True,verbose=False)
                if text_a is None:
                    continue
                # sScore映射成我们需要的, -1，0，1格式
                text = left_text + "$T$" + right_text
                labelid = str(class2id[label])
                if print_example > 0:
                    print(text)
                    print(keyword)
                    print(labelid)
                    print_example -= 1
                total += 1
                f.write(text + "\n")
                f.write(keyword + "\n")
                f.write(labelid + "\n")
        print(f"文件已生成{filename}, 总数据量是{total}")
    save2file(filename=train_file, data=train_data)
    save2file(filename=test_file, data=dev_data)

def components_process_caiyang(train_file, test_file):
    """
    使用采样后的数据
    $T$ is super fast , around anywhere from 35 seconds to 1 minute .
    Boot time
    1
    :param train_file: 0.8的比例存储，
    :param test_file: 0.2
    :param truncate: 截断处理，截断前后，默认为None，可以为int，截断保留数字
    :return: 存储到文件
    """
    from convert_label_studio_data import get_all
    import json
    data = get_all(absa=False, keep_cancel=False, split=False)
    # 原始文件中的sScore的映射方式
    train_read_file = "/Users/admin/git/sentiment_pytorch/datasets/com2/caiyang_data/total_train.txt"
    test_read_file = "/Users/admin/git/sentiment_pytorch/datasets/com2/caiyang_data/total_test.txt"
    def save2file(readfile, filename):
        # 打印多少条样本
        print_example = 10
        # 总数据量
        total = 0
        with open(filename, 'w') as f, open(readfile, 'r') as fread:
            for line in fread:
                one_data = json.loads(line)
                text = one_data['content']
                keyword = one_data['aspectTerm']
                start_idx = one_data['start']
                end_idx = one_data['end']
                label = str(one_data['label'])
                # 使用 $T$代表apsect
                text_a, left_text,right_text = aspect_truncate(text,keyword,start_idx,end_idx)
                # sScore映射成我们需要的, -1，0，1格式
                text = left_text + "$T$" + right_text
                if print_example > 0:
                    print(text)
                    print(keyword)
                    print(label)
                    print_example -= 1
                total += 1
                f.write(text + "\n")
                f.write(keyword + "\n")
                f.write(label + "\n")
        print(f"文件已生成{filename}, 总数据量是{total}")
    save2file(readfile=train_read_file, filename=train_file)
    save2file(readfile=test_read_file, filename=test_file)

def check_data(save_file):
    """
    没啥用，检查下数据
    :param save_file:
    :return:
    """
    with open(save_file, 'r') as f:
        lines = f.readlines()

    without_aspect = []
    contents_lenth = []
    all_aspects = []
    for line in lines:
        line_chinese = json.loads(line)
        if not line_chinese["aspect"]:
            without_aspect.append(line_chinese)
            print(line_chinese)
        else:
            contents_lenth.append(len(line_chinese["content"]))
        for aspect in line_chinese["aspect"]:
            aspectTerm = aspect["aspectTerm"]
            all_aspects.append(aspectTerm)
    print(f"没有aspect的数量是{len(without_aspect)}")
    max_lenth = max(contents_lenth)
    max_aspect = max(map(len, all_aspects))
    max_aspect_word = list(filter(lambda x: len(x)>20, all_aspects))
    print(f"最大的句子长度是{max_lenth}")
    print(f"最长的Apsect长度是{max_aspect}")
    print(f"长度大于20的aspect有{max_aspect_word}")

def clean_cache():
    """
    删除../data/cosmetics/cached* 文件
    :return:
    """
    os.system("rm -rf ../datasets/cosmetics/cached*")
    os.system("rm -rf ../logs/*")

def prepare_for_word2vec(save_file):
    """
    拆分成sentence_file 和user_dict 用于训练词向量
    :param save_file:
    :return: sentence_file, user_dict
    """
    sentence_file = os.path.join(os.path.dirname(save_file), "sentence_file.txt")
    user_dict = os.path.join(os.path.dirname(save_file), "user_dict.txt")
    with open(save_file, 'r') as f:
        lines = f.readlines()
    with open(sentence_file, 'w') as sf:
        with open(user_dict, 'w') as uf:
            for line in lines:
                line_chinese = json.loads(line)
                # 使用 $T$代表apsect
                content = line_chinese["content"]
                sf.write(content + "\n")
                # 如果这个句子没有aspect，那就过滤掉
                if not line_chinese["aspect"]:
                    continue
                for aspect in line_chinese["aspect"]:
                    aspectTerm = aspect["aspectTerm"]
                    uf.write(aspectTerm + "\n")
    return sentence_file, user_dict

def train_word2vec(sentence_file, user_dict, dimension=300):
    """
    word2vec 训练词向量
    :param sentence_file: 原始文件，包含所有语句
    :param user_dict: 用户自定义的字典
    :param dimension: 嵌入维度
    :return:
    """
    import gensim
    from gensim.models import word2vec
    import jieba.analyse
    import jieba
    # 加载自定义词典
    jieba.load_userdict(user_dict)
    # 分隔的单词
    word_file_path = os.path.join(os.path.dirname(sentence_file), "word_file.txt")
    model_path = os.path.join(os.path.dirname(sentence_file),"cosmetics_300d.txt")
    with open(word_file_path, 'w', encoding='utf-8') as writer:
        with open(sentence_file, 'r', encoding='utf-8') as reader:
            # 加载所有数据
            content = reader.read()
            # 分词
            content = jieba.cut(content)
            # 合并结果
            result = ' '.join(content)
            # 结果输出
            writer.write(result)
    # 加载单词
    sentences = word2vec.LineSentence(word_file_path)
    # 训练词向量
    # sg: 1代表(Skip-gram) 0(CBOW)， 默认为0
    # hs: 1代表hierarchical softmax  0代表负采样negative， 默认为0
    model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1,
                              window=3, size=dimension, compute_loss=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=5)
    #保存成纯文本文件
    model.wv.save_word2vec_format(model_path, binary=False)

def conver_embedding_file():
    """
    转换我们自己的预训练词向量，到embdding file的格式为
    单词 300d向量
    :return:
    """
    import numpy as np
    import json
    embedding_file = "../embedding/model.npy"
    word2index_file = "../embedding/word2index.json"
    # 保存新的txt格式的embedding
    word2vec_file = "../embedding/cosmetics_300d_w2v.txt"
    embedding_array = np.load(embedding_file)
    print("embdding array的shape是(单词数，嵌入维度):",embedding_array.shape)

    #但因嵌入单词的索引
    with open(word2index_file,'r',encoding="utf-8") as f:
        content = f.read()
    word_index = json.loads(content)
    print("索引中单词总数是: ",len(word_index))
    print("eg: 单词 [马拉河] 的索引是:",word_index['马拉河'])
    id2index = {v:k for k,v in word_index.items()}
    with open(word2vec_file, 'w', encoding="utf-8") as f:
        for idx, arr in enumerate(embedding_array):
            word = id2index[idx]
            if word == "[ZERO]":
                word = "0"
            string_array = " ".join(map(str, arr.tolist()))
            f.write(f"{word} {string_array}\n")

if __name__ == '__main__':
    # save_file = "../data_root_dir/cosmetics/all.txt"
    # new_file = "../data_root_dir/cosmetics/final_all.txt"
    # db2local(save_file)
    # textsentiment_process(save_file, new_file)
    # split_all(new_file, train_rate=0.9, test_rate=0.1)
    # check_data(save_file)
    # clean_cache()
    # conver_embedding_file()
    # sentence_file, user_dict = prepare_for_word2vec(save_file)
    # train_word2vec(sentence_file, user_dict)
    # components_process_caiyang(train_file='/Users/admin/git/sentiment_pytorch/datasets/components/train.txt', test_file='/Users/admin/git/sentiment_pytorch/datasets/components/test.txt')
    # components_process(train_file='/Users/admin/git/sentiment_pytorch/datasets/components/train.txt', test_file='/Users/admin/git/sentiment_pytorch/datasets/components/test.txt')
    components_process_truancate(train_file='/Users/admin/git/sentiment_pytorch/datasets/components/train.txt', test_file='/Users/admin/git/sentiment_pytorch/datasets/components/test.txt')
