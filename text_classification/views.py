from django.http import JsonResponse
from django.views.generic import View
from d2l import torch as d2l
import torch
import pickle
import json
from utils.classification import predict, predict_doc_visualize
from utils.classification_model import semi_bert


class TextClassification(View):
    p_type = 2

    def post(self, request, *args, **kwargs):
        # 解析 JSON 数据
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if self.p_type:
            text = body_data.get('text_list', [])  # 获取语句列表
            print("Received text list:", text)  # 调试输出
        else:
            text = body_data.get('text')  # 获取单一语句
            print("Received single text:", text)  # 调试输出

        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        device = d2l.try_gpu()

        embed_size = 100  # 词嵌入维度，我选了100维的
        ffn_hiddens = 64  # FFN，隐藏层单元数量
        num_heads = 4  # 注意力头的个数
        num_blks = 1  # transformer_block的个数
        dropout = 0.5  # dropout率（用于正则化）
        max_len = 50  # 每个句子的最大长度

        with open('utils/classification_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

        net = semi_bert(len(vocab), embed_size, ffn_hiddens, num_heads, num_blks, dropout, max_len)
        net.load_state_dict(torch.load('utils/classification.pth', map_location=device))

        if self.p_type:
            print(text, type(text))
            positive_comments_loc, negative_comments_loc = predict_doc_visualize(net, vocab, text)
            head = 'http://127.0.0.1:8000/'
            # 可以生成词云图返回词云图url, 不可以生成词云图返回None
            if positive_comments_loc is not None:
                positive_comments_wordcloud_url = head + 'media/' + positive_comments_loc + '/'
            else:
                positive_comments_wordcloud_url = None

            if negative_comments_loc is not None:
                negative_comments_wordcloud_url = head + 'media/' + negative_comments_loc + '/'
            else:
                negative_comments_wordcloud_url = None

            return JsonResponse({
                'positive_comments_wordcloud_url': positive_comments_wordcloud_url,
                'negative_comments_wordcloud_url': negative_comments_wordcloud_url
            })
        else:
            result = predict(net, vocab, text)
            return JsonResponse({'result': result})
