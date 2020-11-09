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
    save_file = "../data_root_dir/cosmetics/all.txt"
    new_file = "../data_root_dir/cosmetics/final_all.txt"
    # db2local(save_file)
    textsentiment_process(save_file, new_file)
    split_all(new_file, train_rate=0.9, test_rate=0.1)
    # check_data(save_file)
    # clean_cache()
    # conver_embedding_file()
    # sentence_file, user_dict = prepare_for_word2vec(save_file)
    # train_word2vec(sentence_file, user_dict)
