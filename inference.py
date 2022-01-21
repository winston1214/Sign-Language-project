import sys
import os
from model.seq2seq_lstm import LSTM_Encoder,LSTM_Decoder,LSTM_Seq2Seq
from model.seq2seq_gru_attention import GRU_AT_Decoder, GRU_AT_Encoder, GRU_AT_Seq2Seq, Attention
from scripts.demo_inference import alphapose_inference
from preprocessing.frame_split import frame_split
from utils.seq2seq_preprocessing import  target_preprocessing
from utils.train_utils import translate_SL,translate_SL_ATT,init_weights

import argparse
import json
import time
import re
import numpy as np
import torch
import torch.utils.data as D
import torch.backends.cudnn as cudnn
import random
import numpy as np
import shutil

import time
import argparse
import pandas as pd


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)



def inference(opt):
    ### video frame split
    start_time = time.time()
    if os.path.exists('frame'):
        shutil.rmtree('frame')
    os.mkdir('frame')
    indir = 'frame/'
    video_name = opt.video_name.split('/')[-1]
    frame_split(video_name,indir)
    checkpoint = opt.checkpoint
    cfg = opt.cfg
    format = opt.format
    outdir = opt.outdir
    sp = opt.sp
    alphapose_inference(checkpoint,cfg,format,outdir,sp) # indir로 frame 저장 폴더 만들기
    with open('alphapose-results.json','r') as f:
        data = json.load(f)
    max_frame_num = 376
    video_key = np.array([])
    num_ls = []
    for i in data: # 전체 영상 증가 - frame normalization + face remove + reverse

        num = re.sub('.jpg','',i)
        num_ls.append(int(num))

        dt = data[i]['keypoints']
        dt = np.array(dt).reshape(123,3)
        dt = np.delete(dt,range(13,81),axis=0)
        dt = dt[:,:2]
        x,y = dt[:,0],dt[:,1]
        mean_x, mean_y = np.mean(x),np.mean(y)
        std_x,std_y = np.std(x),np.std(y)
        normal_x,normal_y = (x-mean_x)/std_x , (y - mean_y)/std_y
        dt[:,0] = normal_x
        dt[:,1] = normal_y
        video_key = np.append(video_key,dt)
    start_num = min(num_ls)
    video_key = video_key.reshape(-1,110)
    
    if len(data) < max_frame_num:
        random_choice_frame = np.random.choice(num_ls,max_frame_num - max(num_ls) + start_num -1)
        random_choice_frame.sort()
        for random_idx,h in enumerate(random_choice_frame):
            insert_num = h + random_idx - start_num
            video_key = np.insert(video_key,insert_num,video_key[insert_num],axis=0)
    else:
        video_key = video_key[-max_frame_num:]

    video_key = video_key[::-1]
    reverse_video = video_key.copy()
    X_data = torch.tensor(reverse_video)
    X_data = torch.unsqueeze(X_data,0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device : ', device)
    excel_name = opt.csv_name
    word_to_index, _, train_vocab ,_ = target_preprocessing(excel_name,'train')
    input_size = X_data.shape[-1] # keypoint vector 길이
    HID_DIM = opt.hid_dim # 512
    OUTPUT_DIM = len(train_vocab)
    N_LAYERS = 2
    DEC_DROPOUT = opt.dropout # 0.5
    emb_dim = opt.emb_dim # 128

    ## Define Model
    if opt.model == 'LSTM':
        enc = LSTM_Encoder(input_size, HID_DIM, N_LAYERS)
        dec = LSTM_Decoder(OUTPUT_DIM, emb_dim, HID_DIM, N_LAYERS, DEC_DROPOUT)
        model = LSTM_Seq2Seq(enc, dec, device).to(device)

    if opt.model == 'GRU':
        enc = GRU_AT_Encoder(input_size, HID_DIM, N_LAYERS)
        att = Attention(HID_DIM)
        dec = GRU_AT_Decoder(OUTPUT_DIM, emb_dim, HID_DIM, N_LAYERS, att, DEC_DROPOUT)
        model = GRU_AT_Seq2Seq(enc,dec,device).to(device)
    model.apply(init_weights)
    model.load_state_dict(torch.load(opt.pt))


    if opt.model == 'LSTM':
        translate = translate_SL(X_data,word_to_index,model,device)
        
    if opt.model == 'GRU':
        translate = translate_SL_ATT(X_data,word_to_index,model,device)
    sentence = ' '.join(translate)
    print(f'Sign Language Translate : {sentence}')
    end_time = time.time()
    print(f'Time : {end_time - start_time :02} sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign-Language-Train')
    parser.add_argument('--checkpoint',type=str,default='pretrained_models/halpe136_fast_res50_256x192.pth',help = 'download pth')  # No modify
    parser.add_argument('--cfg', type=str, default='configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml',help='cfg')  # No modify
    parser.add_argument('--format',type=str,default='boaz',help='coco,open,cmu,boaz')  # No modify
    parser.add_argument('--video_name',type=str)
    parser.add_argument('--outdir',type=str,default='result/')
    parser.add_argument('--sp', default=True,help = 'if you use multi-gpu, check sp False')
    parser.add_argument('--hid_dim', type=int, default=512,help='Number of hidden demension') # No modify
    parser.add_argument('--dropout',type=float,default=0.5,help = 'dropout ratio') # No modify
    parser.add_argument('--emb_dim',type=int,default=128,help = 'Nuber of embedding demension') # No modify
    parser.add_argument('--csv_name',type=str,default='train_target.csv',help='Target Excel name')
    parser.add_argument('--model',type=str,default='GRU',help='[LSTM,GRU]')
    parser.add_argument('--pt',type=str,default='GRU_TUNNING.pt',help='save model name')
    opt = parser.parse_args()
    inference(opt)
