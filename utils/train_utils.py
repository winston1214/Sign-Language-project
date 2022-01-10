import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import numpy as np
import random
import nltk.translate.bleu_score as bleu


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


def evaluate(model, dataloader,OUTPUT_DIM, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
   
      for i,(input, target) in enumerate(dataloader): # valid_dataloader 정의하기
            src = input
            trg = target

            if torch.cuda.is_available():
                model.cuda()
                src = src.cuda().float()
                trg = trg.cuda()

            output = model(src, trg, 0)
            #trg = [trg len, batch size] [16,12]
            #output = [trg len, batch size, output dim]
            
            output = output[1:].view(-1, OUTPUT_DIM)
            trg = torch.transpose(trg,0,1)
            trg = trg[1:].contiguous().view(-1)
                
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def translate_SL(src, word_to_index, model, device, max_len = 81):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''

    model.eval()

    # sign_language = "<sos>" + sign_language + "<eos>"
    # print(f"sign language: {sign_language}")
    # 인덱스 파트 (변수명: src_tensor)

    with torch.no_grad():

        hidden, cell = model.encoder(src)
  
    trg_indexes = [word_to_index['s']]

    for i in range(max_len):
        # trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        trg_tensor = torch.tensor([trg_indexes[-1]],dtype=torch.long).to(device)
        # print(trg_tensor)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # # <eos>를 만나는 순간 끝
        # if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
        #     break

        # 각 출력 단어 인덱스를 실제 단어로 변환
        # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        # trg_tokens = [word_to_index[i] for i in trg_indexes]
  
  trg_tokens = [list(word_to_index)[i] for i in trg_indexes]


    # trg_tokens = [key for key, value in word_to_index.items() if value == i]


    # 첫 번째 <sos>는 제외하고 출력 문장 반환
  return trg_tokens[1:]

def BLEU_Evaluate(model,dataloader,criterion, word_to_index,OUTPUT_DIM , device, max_len = 12):
    '''
    src: 번역하고자 하는 keypoint
    word_to_index: korean index 뭉치
    '''
    chencherry = bleu.SmoothingFunction()
    model.eval()
    epoch_loss = 0
    for _,(input, target) in enumerate(dataloader): 
        BLEU = 0
        src = input
        trg = target

        if torch.cuda.is_available():
            model.cuda()
            src = src.cuda().float()
            src2 = src
            trg = trg.cuda()
            trg2 = trg

        with torch.no_grad(): # evaluation
            
            hidden, cell = model.encoder(src)
            output = model(src,trg,0) # evaluate
            output = output[1:].view(-1,OUTPUT_DIM) # evaluate
            trg = torch.transpose(trg,0,1) # evaluate
            trg = trg[1:].contiguous().view(-1) # evaluate

            loss = criterion(output,trg) # evaluate
            epoch_loss += loss.item() # evaluate
        
        for input_data,target in zip(src,trg2):
            input_data = torch.unsqueeze(input_data, 0)
            print(target[1])
            ref = list(word_to_index)[target[1]]
            
            candidate = ' '.join(translate_SL(input_data, word_to_index, model, device))
            reg = re.sub('[sf]','',ref)
            # print('candidate',candidate)
            # print('reg',reg)
            BLEU += bleu.sentence_bleu(ref.split(), candidate.split(), weights = (0.5, 0.5), smoothing_function=chencherry.method4, auto_reweigh=False)


    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return BLEU / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
