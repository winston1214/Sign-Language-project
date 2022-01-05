import os
import numpy as np
import json
import pandas as pd
import re
# from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils.pad_sequences import pad_sequences
from utils.keypoint_process import video_sampling
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
excel_name = 'train_target.xlsx'


target = pd.read_excel(excel_name)

# 여기서 추출하는 과정 거쳐야함(원하는 인덱스 비디오만 추출하는 과정)

target = target['target'].map(lambda x: re.sub('[(-,)=.#/?:$}]','', x)) # 부가적인 전처리
target = target.apply(lambda x : 's '+ x + ' f')

temp = target.values.tolist() # 단어 집합 구축


vocab = {}
preprocessed_sentences = []

for sentence in temp:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
      result.append(word)
      if word not in vocab:
        vocab[word] = 0
      vocab[word] += 1 # 단어와 빈도수 집합
    preprocessed_sentences.append(result) 
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)

word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted :
    if frequency > 0 : # 빈도수가 작은 단어는 제외.안해!!!
        i = i + 1 # 왜 index번호가 1부터 들어가는지 체크
        word_to_index[word] = i
'''
수정 필요 -> preprocessed_sentence와 word_to_index 개수가 다름.. 왜?
encoded_sentences = [] # 각 글자에 index 부여
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
      encoded_sentence.append(word_to_index[word])
    encoded_sentences.append(encoded_sentence)
'''
max_tar_len = max(map(lambda x: len(x),temp))
decoder_input = pad_sequences(encoded_sentences, maxlen=max_tar_len, padding='post')
