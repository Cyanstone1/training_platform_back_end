from django.http import JsonResponse
from django.views.generic import View
from gensim.models import Word2Vec
from utils.comparison import read_txt, score_word2vec, similar_count, bert_vec

class TextComparison(View):
    p_type = 2

    def post(self, request, *args, **kwargs):
        if self.p_type:
            text1 = request.POST.get('sentence1')
            text2 = request.POST.get('sentence2')
            vec1 = bert_vec(text1)
            vec2 = bert_vec(text2)
            result = similar_count(vec1, vec2, model='cos')
        else:
            text1 = request.FILES.get('file1')
            text2 = request.FILES.get('file2')
            text_list1 = read_txt(text1)
            text_list2 = read_txt(text2)
            model = Word2Vec.load('utils/file_comparison_model.txt')
            result = score_word2vec(model, text_list1, text_list2)
        
        return JsonResponse({'result': str(result)})
