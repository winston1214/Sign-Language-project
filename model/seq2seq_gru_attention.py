import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

random.seed(0)
np.random.seed(0)


# Encoder 정의하기 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GRU_AT_Encoder(nn.Module):
    def __init__(self, input_size, hid_dim, n_layers):
        super().__init__() 
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # batch_first = True
        self.gru = nn.GRU(input_size, hid_dim, n_layers, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        # input_size = keypoint 개수

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE)임  (다양한 length를 다룰 수 있습니다.).
        # 최초의 hidden state와 cell state를 초기화
        
        # x = ([16, 81, 246])
        # h0 = (2, batch, 512)
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hid_dim).to(device).float()

        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layer * n directions, batch size, hid dim]
        # print(h0.shape) # torch.Size([2, 2, 512])
        # LSTM 순전파
        out, hidden = self.gru(x, h0)
        #print(hidden.shape)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
    
        # print(out.shape) # torch.Size([2, 227, 1024]) = batch, frame, hidden*2

        # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. 
        return out, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        # hidden = (2, batch, hid_dim) 
        # output =  batch, frame, hidden*2

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # print(hidden.shape) # torch.Size([2, 512])


        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        # print(hidden.shape)
        # print(encoder_outputs.shape)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
class GRU_AT_Decoder(nn.Module):
    def __init__(self, output_dim,  emb_dim, hid_dim, n_layers, attention,dropout):
        super().__init__() 

        # output_dim = len(TRG.vocab) softmax로 총 단어 집합 중 어떤 단어인지 선택

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.gru = nn.GRU(hid_dim*2+emb_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim*2)+hid_dim+emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, input, hidden, encoder_outputs): 

        # LSTM 순전파
        # input = [16]
        input = input.unsqueeze(0)  # 1,16
        embedded = self.dropout(self.embedding(input)) # 1, 16, 128

        attention = self.attention(hidden, encoder_outputs)
        attention = attention.unsqueeze(1)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(attention, encoder_outputs)
        # print('weighted.shape :', weighted.shape) # weighted.shape : torch.Size([2, 1, 1024])
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        # print('rnn_input shape: ' , rnn_input.shape) # rnn_input shape:  torch.Size([1, 2, 1152])
        # print('hidden.shape: ', hidden.shape) # hidden.shape:  torch.Size([2, 512])


        out1, hidden = self.gru(rnn_input, hidden.unsqueeze(0)) # # hidden.shape:  torch.Size([1, 2, 512])


        assert (out1 == hidden).all()
        
        embedded = embedded.squeeze(0)
        out1 = out1.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((out1, weighted, embedded), dim = 1))
 

        # 마지막 time step(sequence length)의 hidden state를 사용해 Class들의 logit을 반환합니다(hid_dim -> num_classes). 
        #out1 = self.fc(out1[:, -1, :])
        
        return prediction,hidden.squeeze(0)

class GRU_AT_Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
    
        batch_size = trg.shape[0] # 16 batch_size
        trg_len = trg.shape[1] #12 padding
        trg_vocab_size = self.decoder.output_dim

        # output을 저장할 tensor를 만들기.(처음에는 전부 0으로)
        outputs = torch.zeros(trg_len, batch_size,trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        # outputs = (12(padding), 16(batch), 단어사전의 총 단어개수)
        # src문장을 encoder에 넣은 후 hidden, cell값을 구합니다.

        
        
        # decoder에 입력할 첫번째 input입니다.
        # 첫번째 input은 모두 <sos> token입니다.
        # trg[0,:].shape = BATCH_SIZE 

        # 첫번째인풋은 vocab 모음 중에서 /t를 인코딩한 부분 !!
        input = trg[:,0]

        '''한번에 batch_size만큼의 token들을 독립적으로 계산 -> 즉 frame개수 만큼 Batch 설정??
        즉, 총 trg_len번의 for문이 돌아가며 이 for문이 다 돌아가야지만 하나의 문장이(지금은 단어가) decoding됨
        또한, 1번의 for문당 128개의 문장의 각 token들이 다같이 decoding되는 것'''
        for t in range(1, trg_len): # range(1, 12) -> 12는 단어 하나의 길이를 padding

            #trg_len = 12
            # input token embedding과 이전 hidde state를 decoder에 입력합니다.
            # 새로운 hidde states와 예측 output값이 출력됩니다.

            output, hidden = self.decoder(input, hidden, encoder_outputs)
            #이때 output은 final_state = out1[:,-1,:], prediction = self.fc_out(final_state) 
            #output = [batch_size, output dim] prediction 은?:  torch.Size([1, 341])
            # 각각의 출력값을 outputs tensor에 저장합니다.
            outputs[t] = output
            # outputs = (12(padding), 16(batch), 단어사전의 총 단어개수)
            
            teacher_force = random.random() < teacher_forcing_ratio
      
            # top1 = [batch size]
            top1 = output.argmax(1)

            # teacher forcing기법을 사용한다면, 다음 input으로 target을 입력하고
            # 아니라면 이전 state의 예측된 출력값을 다음 input으로 사용합니다.

            input = trg[:,t] if teacher_force else top1

        return outputs