import json
from django.http import JsonResponse
from django.views.generic import View
from gensim.models import Word2Vec
from utils.comparison import read_txt, score_word2vec, similar_count, bert_vec


class TextComparison(View):
    p_type = 2

    def post(self, request, *args, **kwargs):
        try:
            if self.p_type:
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                text1 = body_data.get('sentence1')
                text2 = body_data.get('sentence2')

                if not text1 or not text2:
                    return JsonResponse({'error': 'Both sentence1 and sentence2 are required.'}, status=400)

                vec1 = bert_vec(text1)
                vec2 = bert_vec(text2)
                result = similar_count(vec1, vec2, model='cos')
            else:
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                file1_text = body_data.get('file1')
                file2_text = body_data.get('file2')

                if not file1_text or not file2_text:
                    return JsonResponse({'error': 'Both file1 and file2 are required.'}, status=400)

                # 模拟读取文件内容
                text_list1 = file1_text.split('\n')
                text_list2 = file2_text.split('\n')

                model = Word2Vec.load('utils/file_comparison_model.txt')
                result = score_word2vec(model, text_list1, text_list2)

            return JsonResponse({'result': str(result)})
        except Exception as e:
            print("Error occurred:", e)
            return JsonResponse({'error': str(e)}, status=500)
