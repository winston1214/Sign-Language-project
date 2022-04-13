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

def german_bleu(data):
    pred = data['predict']
    target = data['answer']
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred.values,target.values,avg=True)['rouge-l']['f'] # ROUGE-L
    pred2 = [p.lower().split() for p in pred]
    target2 = [[t.lower().split()] for t in target]
    final = 0
    meteor = 0
    for a,p,a2,p2 in zip(target,pred,target2,pred2):
        n = len(a2[0][0])
        if n<4:
            w = [1/n] * n + [0] * (4-n)
        else:
            w = [0.25,0.25,0.25,0.25]
        final += sentence_bleu(a2,p2,weights = w)
        meteor += nltk.translate.meteor_score.meteor_score([a],p,gamma=0)
    return final/len(data),rouge_score,meteor/len(data)
