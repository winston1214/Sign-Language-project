# -*- coding: utf-8 -*-
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import numpy as np
import random
import nltk.translate.bleu_score as bleu
from nltk.metrics import scores
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from metric.bleu import compute_bleu
warnings.filterwarnings('ignore')

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train(model, dataloader, OUTPUT_DIM,optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
   
    for i, (input, target) in enumerate(dataloader):

        src = input
        trg = target

        if torch.cuda.is_available():
            model.cuda()
            src = src.cuda().float()
            trg = trg.cuda()
       
        # src = [16, 81, 246] batch, frame수, keypoint수
        # trg(trg)= [16, 12] = batch, trg_len

        optimizer.zero_grad()        
        
        output = model(src, trg)
        #trg = [trg len, batch size] [16,12]
        #output = [trg len, batch size, output dim]
        
        output = output[1:].view(-1, OUTPUT_DIM)
        trg = torch.transpose(trg,0,1)
        trg = trg[1:].contiguous().view(-1)
              
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)



def translate_SL(src, word_to_index, model, device, max_len = 81):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''
    if torch.cuda.is_available():
        model.cuda()
        src = src.cuda().float()

    model.eval()

    # sign_language = "<sos>" + sign_language + "<eos>"
    # print(f"sign language: {sign_language}")
    # 인덱스 파트 (변수명: src_tensor)

    with torch.no_grad():

        hidden, cell = model.encoder(src)
  
    trg_indexes = [word_to_index['s']]
    end_index = word_to_index['f']
    for _ in range(max_len):
        # trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        trg_tensor = torch.tensor([trg_indexes[-1]],dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        # # <eos>를 만나는 순간 끝
        if pred_token == end_index:
            break
        trg_indexes.append(pred_token) # 출력 문장에 더하기



        # 각 출력 단어 인덱스를 실제 단어로 변환
        # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        # trg_tokens = [word_to_index[i] for i in trg_indexes]
  
    trg_tokens = [list(word_to_index)[i] for i in trg_indexes]


    # trg_tokens = [key for key, value in word_to_index.items() if value == i]


    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:]

def translate_SL_ATT(src, word_to_index, model, device, max_len = 81):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''
    if torch.cuda.is_available():
        model.cuda()
        src = src.cuda().float()
    model.eval()

  # sign_language = "<sos>" + sign_language + "<eos>"
  # print(f"sign language: {sign_language}")
  # 인덱스 파트 (변수명: src_tensor)

    with torch.no_grad():

        encoder_outputs, hidden = model.encoder(src)
  
    trg_indexes = [word_to_index['s']]
    end_index = word_to_index['f']


    for i in range(max_len):
        # trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        trg_tensor = torch.tensor([trg_indexes[-1]],dtype=torch.long).to(device)

        with torch.no_grad():
            output, hidden= model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
            # # <eos>를 만나는 순간 끝
        if pred_token == end_index:
            break

        trg_indexes.append(pred_token) # 출력 문장에 더하기




    # 각 출력 단어 인덱스를 실제 단어로 변환
    # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    # trg_tokens = [word_to_index[i] for i in trg_indexes]
    trg_tokens = [list(word_to_index)[i] for i in trg_indexes]
    # trg_tokens = [key for key, value in word_to_index.items() if value == i]


    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:]

def BLEU_Evaluate(model,dataloader,criterion, word_to_index,OUTPUT_DIM , device, max_len = 81,model_name = 'GRU'):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''
    zero_pred = 0
    model.eval()
    epoch_loss = 0
    BLEU = 0
    acc = 0
    cnt = 0
    for _,(input, target) in enumerate(dataloader): 
        
        src = input
        trg = target

        if torch.cuda.is_available():
            model.cuda()
            src = src.cuda().float()

            trg = trg.cuda()
            trg2 = trg

        with torch.no_grad(): # evaluation
            
            # hidden, cell = model.encoder(src)
            output = model(src,trg,0) # evaluate
            output = output[1:].view(-1,OUTPUT_DIM) # evaluate
            trg = torch.transpose(trg,0,1) # evaluate
            trg = trg[1:].contiguous().view(-1) # evaluate

            loss = criterion(output,trg) # evaluate
            epoch_loss += loss.item() # evaluate
        
        for input_data,target in zip(src,trg2):
            input_data = torch.unsqueeze(input_data, 0)
            cnt += 1
            ref = []
            for t in target:
                if t == 1:
                    break
                else:
                    ref.append(list(word_to_index)[t])
            ref = ' '.join(ref)
            if model_name == 'GRU':
                candidate = ' '.join(translate_SL_ATT(input_data, word_to_index, model, device,max_len))
            else:
                candidate = ' '.join(translate_SL(input_data, word_to_index, model, device,max_len))
            ref = re.sub('[sf]','',ref)
            ref = ref.strip()
            if len(candidate.split()) == 0:
                zero_pred += 1
            else:
                BLEU += bleu.sentence_bleu([ref.split()], candidate.split(),weights = [1,0,0,0])
                acc += sum(x == y for x, y in zip(ref.split(), candidate.split())) / len(candidate.split())
    if cnt == zero_pred:
        BLEU , acc = 0,0
    else:
        BLEU = BLEU / (cnt - zero_pred)
        acc = acc/(cnt - zero_pred)
    print('zero : ',zero_pred)
    return epoch_loss / len(dataloader), BLEU, acc


def BLEU_Evaluate_test(model,dataloader,word_to_index, word_to_index_test, device , max_len = 81,model_name = 'GRU'):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''
    model.eval()
    BLEU = 0
    acc = 0
    cnt = 0
    answer = []
    predict = []
    zero_pred = 0
    for _,(input, target) in enumerate(dataloader): 
        
        src = input
        trg = target

        if torch.cuda.is_available():
            model.cuda()
            src = src.cuda().float()

            trg = trg.cuda()

        
        for input_data,target in zip(src,trg): # 배치
            input_data = torch.unsqueeze(input_data, 0)
            cnt += 1
            ref = []
            for t in target:

                if t == 1: # f가 나올때까지
                    break
                else:
                    ref.append(list(word_to_index_test)[t])

            ref = ' '.join(ref) # 정답

            if model_name == 'GRU':
                candidate = ' '.join(translate_SL_ATT(input_data, word_to_index, model, device,max_len))
            else:
                candidate = ' '.join(translate_SL(input_data, word_to_index, model, device,max_len))
            
            ref = re.sub('[sf]','',ref)
            ref = ref.strip()

            answer.append(ref)
            predict.append(candidate)

            if len(candidate.split()) == 0:
                zero_pred += 1
            else:
                BLEU += compute_bleu([ref.split()], candidate.split())[0]
                acc += sum(x == y for x, y in zip(ref.split(), candidate.split())) / len(candidate.split())
        
    print('zero : ',zero_pred)
    return BLEU / (cnt-zero_pred), acc/(cnt-zero_pred),answer,predict

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def make_plot(epoch,score,title,path):
    plt.figure(figsize=  (12,8))
    sns.lineplot(x = epoch,y = score)
    plt.title(title,fontsize = 15)
    plt.savefig(f'{path}{title}.png')
