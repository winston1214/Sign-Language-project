from konlpy.tag import Mecab
from nltk.translate.bleu_score import sentence_bleu

def korean_bleu(data):
    mecab = Mecab()
    ans = data['answer']
    pred = data['predict']
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
    return final_bleu/len(data)
def german_bleu(data):
    pred = data['predict']
    target = data['answer']
    pred2 = [p.lower().split() for p in pred]
    target2 = [[t.lower().split()] for t in target]
    final = 0
    for a,p,a2,p2 in zip(target,pred,target2,pred2):
        n = len(a2[0][0])
        if n<4:
            w = [1/n] * n + [0] * (4-n)
        else:
            w = [0.25,0.25,0.25,0.25]
        final += sentence_bleu(a2,p2,weights = w)
    return final/len(data)

