import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import euclidean_distances  # 欧氏距离
from sklearn.metrics.pairwise import cosine_similarity  # 余弦距离

# 路径需要根据你的项目结构调整
model_path = 'utils/chinese-bert-wwm-ext'


def read_txt(uploaded_file):
    f = uploaded_file.readlines()
    temp_txt = []

    for i in f:
        if str(i) == str(b'\n'):  # 判断空行
            continue
        if len(i) <= 2:  # 判断短行
            continue
        i = i.decode('utf-8').strip()  # 将字节对象解码为字符串，并去除两端的空白字符
        temp_txt.append(i)

    return temp_txt


# TODO: 修改score_word2vec逻辑
def score_word2vec(model, file1, file2):
    '''
    比较两个文件的相似度
    '''
    score = 0
    n = min(len(file1), len(file2))

    for i in range(n):
        if len(file1[i]) < 2 or len(file2[i]) < 2:
            tmp = 0
        else:
            tmp = model.wv.n_similarity(file1[i], file2[i])
        score += tmp

    return score / n


# 从本地路径加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)  # 定义分词器
bert_model = BertModel.from_pretrained(model_path)  # 定义模型


def similar_count(vec1, vec2, model="cos"):
    """
    计算距离
    :param vec1: 句向量1
    :param vec2: 句向量2
    :param model: 用欧氏距离还是余弦距离
    :return: 返回的是两个向量的距离得分
    """
    if model == "eu":
        return euclidean_distances([vec1, vec2])[0][1]
    if model == "cos":
        return cosine_similarity([vec1, vec2])[0][1]
    return None


def bert_vec(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    # 使用这个分词器进行分词
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 把上边的文字转化为词汇表中对应的索引数字
    batch_tokenized = tokenizer.batch_encode_plus([text], padding=True, truncation=True, max_length=20)
    # 最大长度是20 那么超过的就会被截断 不到20的 会将所有的句子补齐到 句子中的最大长度。
    # 1. encode仅返回input_ids
    # 2. encode_plus返回所有的编码信息，具体如下：
    # ’input_ids:是单词在词典中的编码
    # ‘token_type_ids’:区分两个句子的编码（上句全为0，下句全为1）
    # ‘attention_mask’:指定对哪些词进行self-Attention操作
    input_ids = torch.tensor(batch_tokenized['input_ids'])
    attention_mask = torch.tensor(batch_tokenized['attention_mask'])
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    bert_cls_hidden_state = bert_output[0][:, 0, :]
    return np.array(bert_cls_hidden_state[0].detach().numpy())
