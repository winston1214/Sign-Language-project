from konlpy.tag import Mecab
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
def korean_metric(data):
    nltk.download('wordnet')
    mecab = Mecab()
    rouge = Rouge()
    ans = data['answer']
    pred = data['predict']
    rouge_score = rouge.get_scores(pred.values,ans.values,avg=True)['rouge-l']['f'] # ROUGE-L
    meteor = 0
    final_bleu = 0
    for a,p in zip(ans,pred):
        ref = [mecab.morphs(a)]
        candidate = mecab.morphs(p)
        n = len(ref[0])
        if len(ref[0])<4:
            w = [1/n] * n + [0] * (4-n)
        else:
            w = [0.25,0.25,0.25,0.25]
        final_bleu+= sentence_bleu(ref,candidate,weights=w)

        meteor += nltk.translate.meteor_score.meteor_score([a],p,gamma=0)
    return final_bleu/len(data),rouge_score,meteor/len(data)
