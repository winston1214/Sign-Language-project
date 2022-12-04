# -*- coding: utf-8 -*-
from model.seq2seq_lstm import LSTM_Encoder,LSTM_Decoder,LSTM_Seq2Seq
from model.seq2seq_gru_attention import GRU_AT_Decoder, GRU_AT_Encoder, GRU_AT_Seq2Seq, Attention
from utils.seq2seq_preprocessing import  target_preprocessing
from utils.train_utils import BLEU_Evaluate_test,init_weights
import torch
import torch.utils.data as D
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
import numpy as np
import gzip,pickle


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


def main_test(opt):
    
    ### Data Loading
    with gzip.open(opt.X_path + 'X_test.pickle','rb') as f:
        X_data = pickle.load(f)
    excel_name = 'test.csv' # 'C:/Users/winst/Downloads/menmen/train_target.xlsx'
    train_excel_name = 'train.csv'
    if opt.mode == 'asl':
        excel_name  = 'asl_ann/' + excel_name
        train_excel_name = 'asl_ann/' + train_excel_name
    else:
        excel_name  = 'keti_ann/' + excel_name
        train_excel_name = 'keti_ann/' + train_excel_name
    word_to_index_test, max_len, _,decoder_input = target_preprocessing(excel_name,opt.mode)
    word_to_index, _, train_vocab ,_ = target_preprocessing(train_excel_name,opt.mode)

    ## Setting of Hyperparameter
    HID_DIM = opt.hid_dim # 512
    OUTPUT_DIM = len(train_vocab)
    N_LAYERS = 2
    DEC_DROPOUT = opt.dropout # 0.5
    emb_dim = opt.emb_dim # 128
    BATCH_SIZE = opt.batch # 32
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device : ', device)

    ## Change data type
    X_test = torch.tensor(X_data)
    decoder_input = torch.tensor(decoder_input, dtype=torch.long)
    

    dataset = D.TensorDataset(X_test,decoder_input)

    test_dataloader =  torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    input_size = X_test.shape[-1] # keypoint vector 길이

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
    


    ## Test


    start_time = time.time()


    BLEU,acc,answer,predict = BLEU_Evaluate_test(model,test_dataloader, word_to_index,word_to_index_test, device, max_len,model_name = opt.model)
    end_time = time.time()
    result = pd.DataFrame(columns = ['answer','predict'])
    result['answer'] = answer
    result['predict'] = predict

    result.to_csv(f'{opt.save_csv}',index=False)
    print('---save---')
    print(f'Test Time : {end_time - start_time : .3f}')
    print(f'\t TEST BLEU : {BLEU : .3f} | TEST ACC : {acc : .3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign-Language-Test')
    parser.add_argument('--X_path',type=str,default='./',help = 'X_test.pikcle path')
    parser.add_argument('--hid_dim', type=int, default=512,help='Number of hidden demension')
    parser.add_argument('--dropout',type=float,default=0.5,help = 'dropout ratio')
    parser.add_argument('--emb_dim',type=int,default=128,help = 'Nuber of embedding demension')
    parser.add_argument('--batch',type=int,default = 32,help='BATCH SIZE')
    parser.add_argument('--pt',type=str,default='model1.pt',help='save model name')
    parser.add_argument('--csv_name',type=str,default='train_target.csv',help='Target Excel name')
    parser.add_argument('--model',type=str,default='GRU',help='[LSTM,GRU]')
    parser.add_argument('--save_csv',type=str,default = './result.csv',help = 'save result csv name')
    parser.add_argument('--mode',type = str,default='asl')
    opt = parser.parse_args()
    main_test(opt)
