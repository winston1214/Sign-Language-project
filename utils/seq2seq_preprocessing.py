# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.pad_sequences import pad_sequences

import nltk
from nltk.tokenize import word_tokenize
from konlpy.tag import Mecab


def target_preprocessing(excel_name,mode='asl'):
    
    nltk.download('punkt')

    target = pd.read_csv(excel_name)
    
    # 여기서 추출하는 과정 거쳐야함(원하는 인덱스 비디오만 추출하는 과정)
#     target = target.sort_values('num').reset_index(drop=True)
    if mode == 'asl':
        target = target['translation'].map(lambda x: re.sub('[(-,)=.#/?:!$}]','', x)) # 부가적인 전처리
    else:
        target = target['target'].map(lambda x: re.sub('[(-,)=.#/?:!$}]','', x)) # 부가적인 전처리
    target = target.apply(lambda x : 's '+ x + ' f')

    temp = target.values.tolist() # 단어 집합 구축


    vocab = {}
    preprocessed_sentences = []

    for sentence in temp:
        # 단어 토큰화
        if mode == 'asl':   
            tokenized_sentence = word_tokenize(sentence)
        else:
            mecab = Mecab()
            tokenized_sentence = mecab.morphs(sentence)
        result = []

        for word in tokenized_sentence:
            result.append(word)
            if word not in vocab:
                vocab[word] = 0
            else:
                vocab[word] += 1 # 단어와 빈도수 집합
        preprocessed_sentences.append(result) 
    vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)

    word_to_index = {}
    
    for i,(word, frequency) in enumerate(vocab_sorted) :

        word_to_index[word] = i


    encoded_sentences = [] # 각 글자에 index 부여
    for sentence in preprocessed_sentences:
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.append(word_to_index[word])
        encoded_sentences.append(encoded_sentence)

    max_tar_len = max(map(lambda x: len(x),temp))
    decoder_input = pad_sequences(encoded_sentences, maxlen=max_tar_len, padding='post')

    return word_to_index,max_tar_len,vocab,decoder_input
# if __name__ == '__main__':
#     word_to_index,max_tar_len,vocab,decoder_input = target_preprocessing('../asl_train_target.csv')
#     print('vocab',vocab)
#     print('decoder_input',decoder_input.shape)
