# -*- coding: utf-8 -*-
from model.seq2seq_lstm import LSTM_Encoder,LSTM_Decoder,LSTM_Seq2Seq
from model.seq2seq_gru_attention import GRU_AT_Decoder, GRU_AT_Encoder, GRU_AT_Seq2Seq, Attention
from utils.seq2seq_preprocessing import  target_preprocessing
from utils.train_utils import train,BLEU_Evaluate,epoch_time,init_weights,make_plot,BLEU_Evaluate_test,evaluate
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.backends.cudnn as cudnn
import random
import numpy as np
import gzip,pickle
import math
import time
import argparse
from tqdm.notebook import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def main_train(opt):
    
    ### Data Loading
    with gzip.open(opt.X_path + 'X_train.pickle','rb') as f:
        X_data = pickle.load(f)
    train_excel_name = 'train.csv' # 'C:/Users/winst/Downloads/menmen/train_target.xlsx'
    val_excel_name = 'valid.csv'
    if opt.mode == 'asl':
        train_excel_name = 'asl_ann/' + train_excel_name
        val_excel_name = 'asl_ann/' + val_excel_name
    else:
        train_excel_name = 'keti_ann/' + train_excel_name
        val_excel_name = 'keti_ann/' + val_excel_name

    word_to_index, max_len, vocab,decoder_input = target_preprocessing(train_excel_name,mode = opt.mode)

    with gzip.open(opt.X_path + 'X_val.pickle','rb') as f:
        X_val = pickle.load(f)
    val_word_to_index, val_max_len, val_vocab,val_decoder_input = target_preprocessing(val_excel_name,mode = opt.mode)

    ## Setting of Hyperparameter
    HID_DIM = opt.hid_dim # 512
    OUTPUT_DIM = len(vocab)
    N_LAYERS = 2
    DEC_DROPOUT = opt.dropout # 0.5
    emb_dim = opt.emb_dim # 128
    BATCH_SIZE = opt.batch # 32
    N_EPOCHS = opt.epochs # 50
    CLIP = 1
    learning_rate = opt.lr # 0.001
    model_save_path = opt.save_path # 'pt_file/'
    save_model_name = opt.pt_name # 'model1.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device : ', device)

    ## Change data type
    X_train = torch.tensor(X_data)
    X_val = torch.tensor(X_val)

    decoder_input = torch.tensor(decoder_input, dtype=torch.long)
    val_decoder_input = torch.tensor(val_decoder_input,dtype=torch.long)
    print(X_train.shape)
    print(decoder_input.shape)
    print(OUTPUT_DIM)
    train_dataset = D.TensorDataset(X_train,decoder_input)
    val_dataset = D.TensorDataset(X_val,val_decoder_input)
#     train_dataset, val_dataset = D.random_split(dataset, [len(dataset) - int(len(dataset) * 0.2), int(len(dataset) * 0.2)]) # 8:2 split
    train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    input_size = X_data.shape[-1] # keypoint vector 길이 default = 246 


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

    ## Loss & Optimizer

    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    if opt.tunning:
        optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 10,eta_min = 1e-3)


    ## Train

    best_valid_loss = float('inf')
    best_bleu = 0
    best_acc = 0
    BLEU_ls = []
    train_loss_ls = []
    val_loss_ls = []
    acc_ls = []
    epoch_ls = [i for i in range(N_EPOCHS)]
    
    for epoch in tqdm(range(N_EPOCHS)):
        start_time = time.time()

        train_loss = train(model, train_dataloader, OUTPUT_DIM, optimizer, criterion, CLIP)
        BLEU,acc,answer,predict = BLEU_Evaluate_test(model,val_dataloader, word_to_index,val_word_to_index, device,max_len,model_name = opt.model)
        valid_loss = evaluate(model, val_dataloader, OUTPUT_DIM,criterion)
        if opt.tunning:
            scheduler.step()
        train_loss_ls.append(train_loss)
        val_loss_ls.append(valid_loss)
        BLEU_ls.append(BLEU)
        acc_ls.append(acc)
        # valid_loss = evaluate(model, val_dataloader, OUTPUT_DIM,criterion)
 
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_save_path}loss_{save_model_name}')
        if BLEU > best_bleu:
            best_bleu = BLEU
            torch.save(model.state_dict(),f'{model_save_path}bleu_{save_model_name}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),f'{model_save_path}acc_{save_model_name}')
        torch.cuda.empty_cache()
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Val BLEU : {BLEU : .3f} | Val Accuracy : {acc : .3f}')
    with open(f'{model_save_path}best_result.txt','w') as f:
        f.write(f'BLEU : {best_bleu : .3f} \t ACC : {best_acc : .3f}')
    make_plot(epoch_ls,train_loss_ls,'Train loss',opt.save_path)
    make_plot(epoch_ls,val_loss_ls,'Valid loss',opt.save_path)
    make_plot(epoch_ls,BLEU_ls,'BLEU',opt.save_path)
    make_plot(epoch_ls,acc_ls,'Accuracy',opt.save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign-Language-Train')
    parser.add_argument('--X_path',type=str,default='./',help = 'X_train.pikcle path')
    parser.add_argument('--hid_dim', type=int, default=512,help='Number of hidden demension')
    parser.add_argument('--dropout',type=float,default=0.5,help = 'dropout ratio')
    parser.add_argument('--emb_dim',type=int,default=128,help = 'Nuber of embedding demension')
    parser.add_argument('--batch',type=int,default = 32,help='BATCH SIZE')
    parser.add_argument('--epochs',type=int, default = 50, help='EPOCH')
    parser.add_argument('--lr',type=float,default = 0.001,help='learning rate')
    parser.add_argument('--save_path',type=str,default='pt_file',help='model save path')
    parser.add_argument('--pt_name',type=str,default='model1.pt',help='save model name')
    parser.add_argument('--model',type=str,default='GRU',help='[LSTM,GRU]')
    parser.add_argument('--tunning',type=bool,default=False)
    parser.add_argument('--mode',type=str,default='asl')
    opt = parser.parse_args()
    print(opt)
    main_train(opt)
