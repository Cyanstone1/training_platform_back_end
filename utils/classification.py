import os

from d2l import torch as d2l
import torch
import jieba
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

lable_map = ['负向评论', '正向评论']
max_len = 50


def predict(net, vocab, comment):
    '''
    net: 预训练模型
    vocab: 模型训练过程中用到的词典
    comment: 句子
    '''
    print('in predict comment: ', comment)
    net.to(d2l.try_gpu())
    net.eval()
    comment = vocab[[i for i in jieba.cut(comment, cut_all=False)]]
    #将句子进行分词
    comment_pt = torch.tensor(d2l.truncate_pad(comment, max_len, vocab['<unk>']), device=d2l.try_gpu())
    label = torch.argmax(net(comment_pt.reshape(1, -1)), dim=1)
    return lable_map[label]


def predict_doc_visualize(net, vocab, comments):
    """
    net: 预训练模型
    vocab: 模型训练过程中用到的词典
    comments: 句子列表
    """
    net.to(d2l.try_gpu())
    net.eval()
    print('comments', comments)

    # 对评论进行分词,cut_all为分词模式
    commentss = [vocab[[i for i in jieba.cut(j, cut_all=False)]] for j in comments]
    comment_pt = [torch.tensor(d2l.truncate_pad(i, max_len, vocab['<unk>']), device=d2l.try_gpu()) for i in commentss]
    # 预测出来的情感
    labels = [torch.argmax(net(i.reshape(1, -1)), dim=1) for i in comment_pt]

    # def generate_wordcloud(comments, sentiment):
    #     print('text', comments)
    #     text = " ".join(comments)
    #
    #     # font_path是字体文件，需要自己下载
    #     wordcloud = WordCloud(background_color="white", font_path="utils/字体家AI造字特隶.ttf").generate(text)
    #     plt.figure(figsize=(8, 8), facecolor=None)
    #     plt.imshow(wordcloud)
    #     plt.axis("off")
    #     plt.tight_layout(pad=0)
    #     plt.title(f"Word Cloud - {sentiment} Sentiment")
    #
    #     file_name = str(int(time.time())) + '.png'
    #     plt.savefig('media/' + file_name)
    #     return file_name

    def generate_wordcloud(comments, sentiment):
        print('text', comments)
        text = " ".join(comments)

        # font_path是字体文件，需要自己下载
        wordcloud = WordCloud(background_color="white", font_path="utils/字体家AI造字特隶.ttf").generate(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title(f"Word Cloud - {sentiment} Sentiment")

        # 确保media目录存在
        directory = 'media'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = str(int(time.time())) + '.png'
        file_path = os.path.join(directory, file_name)
        plt.savefig(file_path)
        plt.close()
        return file_name

    # 正向评论
    positive_comments = []
    # 负向评论
    negative_comments = []
    for i, label in enumerate(labels):
        if label == 1:
            positive_comments.append(comments[i])
        else:
            negative_comments.append(comments[i])
    if len(positive_comments):
        positive_comments_wordcloud_loc = generate_wordcloud(positive_comments, "Positive")
    else:
        positive_comments_wordcloud_loc = None
    if len(negative_comments):
        negative_comments_wordcloud_loc = generate_wordcloud(negative_comments, "Negative")
    else:
        negative_comments_wordcloud_loc = None
    return positive_comments_wordcloud_loc, negative_comments_wordcloud_loc
