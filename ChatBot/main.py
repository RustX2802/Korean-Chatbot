import math
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext
import torch.optim as optim

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class PositionalEncoder(nn.Module):
    '''입력된 단어의 위치를 원백터정보에 더한다'''
    def __init__(self, position, d_model):
        super().__init__()

        self.d_model = d_model  # 단어 백터의 원래 차원 수

        # 입력 문장에서의 임베딩 벡터의 위치（pos）임베딩 벡터 내의 차원의 인덱스（i）
        pe = torch.zeros(position, d_model)

        # 학습시에는 GPU 사용
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(position):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # pe의 선두에 미니배치 차원을 추가한다
        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False

    def forward(self, x):
        # 입력x와 Positonal Encoding을 더한다
        ret = math.sqrt(self.d_model)*x + self.pe[:, :x.size(1)]
        return ret

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

    depth = key.shape[-1]
    logits = matmul_qk / math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = F.softmax(logits, dim=-1)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        # d_model을 num_heads로 나눈 값.
        self.depth = int(d_model/self.num_heads)

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        # WO에 해당하는 밀집층 정의
        self.out = nn.Linear(d_model, d_model)


    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
      inputs = torch.reshape(
          inputs, (batch_size, -1, self.num_heads, self.depth))
      return torch.transpose(inputs, 1,2)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)


        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)


        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = torch.transpose(scaled_attention, 1,2)

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.out(concat_attention)
        return outputs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, attention):
        outputs = self.linear_1(attention)
        outputs = F.relu(outputs)
        outputs = self.linear_2(outputs)
        return outputs

class EncoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(EncoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)

  def forward(self, inputs, padding_mask):
    attention = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
    attention = self.dropout_1(attention)
    attention = self.norm_1(inputs + attention)
    outputs = self.ff(attention)
    outputs = self.dropout_2(outputs)
    outputs = self.norm_2(attention + outputs)

    return outputs

class Encoder(nn.Module):
  def __init__(self,text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.encoder_block = EncoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, x, padding_mask):
    emb = self.embb(x)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)

    for i in range(self.num_layers):
      output = self.encoder_block(output, padding_mask)

    return output

class DecoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(DecoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.attn_2 = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)
    self.norm_3 = nn.LayerNorm(d_model)

  def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
    attention1 = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
    attention1 = self.norm_1(inputs + attention1)
    attention2 = self.attn_2({'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
    attention2 = self.dropout_1(attention2)
    attention2 = self.norm_2(attention1 + attention2)

    outputs = self.ff(attention2)
    outputs = self.dropout_3(outputs)
    outputs = self.norm_3(attention2 + outputs)

    return outputs

class Decoder(nn.Module):
  def __init__(self,text_embedding_vectors,  vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.decoder_block = DecoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):
    emb = self.embb(dec_input)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)
    for i in range(self.num_layers):
      output = self.decoder_block(output, enc_output, padding_mask, look_ahead_mask)

    return output

class transformer(nn.Module):
    def __init__(self, text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
        self.vocab_size = vocab_size
        super(transformer, self).__init__()
        self.enc_outputs = Encoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.dec_outputs = Decoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.output = nn.Linear(d_model, text_embedding_vectors)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, dec_input):
        enc_input = input
        dec_input = dec_input
        enc_padding_mask = create_padding_mask(enc_input)
        dec_padding_mask = create_padding_mask(enc_input)
        look_ahead_mask = create_look_ahead_mask(dec_input)
    
        enc_output = self.enc_outputs(enc_input, enc_padding_mask)
        dec_output = self.dec_outputs(enc_output, dec_input, dec_padding_mask, look_ahead_mask)
        output = self.output(dec_output)
        return output

import pandas as pd
import re
import urllib.request
import time
import os
from torchtext.legacy import data, datasets
from soynlp.tokenizer import LTokenizer

file_list = []
file_list.append(os.path.join("/root/virtualenvironment/RustX/bin/ChatBot/ChatbotData.csv"))
train_data = pd.read_csv(file_list[0])

tokenizer = LTokenizer()
tokenizer("내일 역 앞의 식당에서 밥 먹으러 나갈래 ?")

VOCAB_SIZE = 40

Q = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=VOCAB_SIZE
)

A = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=VOCAB_SIZE
)

trainset = data.TabularDataset(
    path=file_list[0], format='csv', skip_header=False,
    fields=[('Q', Q),('A', A)])

print(vars(trainset[2]))
print('훈련 샘플의 개수 : {}'.format(len(trainset)))

Q.build_vocab(trainset.Q, trainset.A, min_freq = 2) # 단어 집합 생성
A.vocab = Q.vocab# 단어 집합 생성

PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = Q.vocab.stoi['<pad>'], Q.vocab.stoi['<SOS>'], Q.vocab.stoi['<EOS>'], Q.vocab.stoi['<unk>']

#Define HyperParameter
VOCAB_SIZE = VOCAB_SIZE
text_embedding_vectors = len(Q.vocab)
NUM_LAYERS = 4
D_FF = 512
D_MODEL = 128
NUM_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE=64

# Define Iterator
# train_iter batch has text and target item
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter = data.BucketIterator(
        trainset, batch_size=BATCH_SIZE,
        shuffle=True, repeat=False, sort=False, device = device)
    
# 모델 구축
print(text_embedding_vectors)
net = transformer(text_embedding_vectors = text_embedding_vectors, 
                  vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, d_ff=D_FF, d_model=D_MODEL, 
                  num_heads=NUM_HEADS, dropout=DROPOUT)

# 네트워크 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# 훈련 모드 설정
net.train()

# TransformerBlock모듈의 초기화 설정
net.apply(weights_init)


print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
learning_rate = 2e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def create_padding_mask(x):
  input_pad = 0
  mask = (x == input_pad).float()
  mask = mask.unsqueeze(1).unsqueeze(1)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask

def create_look_ahead_mask(x):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  seq_len = x.shape[1]
  look_ahead_mask = torch.ones(seq_len, seq_len)
  look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1).to(device)

  padding_mask = create_padding_mask(x).to(device) # 패딩 마스크도 포함
  return torch.maximum(look_ahead_mask, padding_mask)

import datetime
# 학습 정의
def train_model(net, train_iter, criterion, optimizer, num_epochs):
    start_time = time.time()

    ntokens = len(Q.vocab.stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)
    print('-----start-------')
    net.to(device)
    epoch_ = []
    epoch_train_loss = []
    # 네트워크가 어느정도 고정되면 고속화
    torch.backends.cudnn.benchmark = True
    
    net.train()
    # epoch 루프
    best_epoch_loss = float("inf")
    for epoch in range(num_epochs):
      epoch_loss = 0.0
      cnt= 0
      for batch in train_iter:
          questions = batch.Q.to(device)
          answers = batch.A.to(device)
          with torch.set_grad_enabled(True):
            # Transformer에 입력
            preds = net(questions, answers)
            pad = torch.LongTensor(answers.size(0), 1).fill_(PAD_TOKEN).to(device)
            preds_id = torch.transpose(preds,1,2)
            outputs = torch.cat((answers[:, 1:], pad), -1)
            optimizer.zero_grad()
            loss = criterion(preds_id, outputs)  # loss 계산
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            epoch_loss +=loss.item()
            cnt += 1
      epoch_loss = epoch_loss / cnt
      if not best_epoch_loss or epoch_loss < best_epoch_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(net.state_dict(), './snapshot/transformermodel.pt')
      
      epoch_.append(epoch)
      epoch_train_loss.append(epoch_loss)
      print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, num_epochs, epoch_loss))
    
    
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()

    ax.plot(epoch_,epoch_train_loss, label='Average loss')


    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    plt.show()
    end_time = time.time() - start_time
    times = str(datetime.timedelta(seconds=end_time)).split(".")
    print('Finished in {0}'.format(times[0]))

num_epochs = 100
#train_model(net, train_iter, criterion, optimizer, num_epochs=num_epochs)

net_trained = transformer(text_embedding_vectors = text_embedding_vectors, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, d_ff=D_FF, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT).to(device)
net_trained.load_state_dict(torch.load('./snapshot/transformermodel.pt'))

def stoi(vocab, token, max_len):
  #
  indices=[]
  token.extend(['<pad>'] * (max_len - len(token)))
  for string in token:
    if string in vocab:
      i = vocab.index(string)
    else:
      i = 0
    indices.append(i)
  return torch.LongTensor(indices).unsqueeze(0)

def itos(vocab, indices):
  text = []
  for i in indices.cpu()[0]:
    if i==1:
      break
    else:
      if i not in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
          if i != UNK_TOKEN:
              text.append(vocab[i])
          else:
              text.append('??')
  return " ".join(text)

def evaluate(input_sentence):
    VOCAB_SIZE = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LTokenizer()
    token = tokenizer(input_sentence)
    input = stoi(Q.vocab.itos, token, VOCAB_SIZE).to(device)
    output = torch.LongTensor(1, 1).fill_(START_TOKEN).to(device)
    for i in range(VOCAB_SIZE):
        predictions = net_trained(input, output)
        predictions = predictions[:, -1:, :]
                            
        #                                      PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions[:,:,3:], axis=-1) + 3
        if predicted_id == END_TOKEN:
            predicted_id = predicted_id
            break
        output = torch.cat((output, predicted_id),-1)
    return output

def predict(sentence):
  out = evaluate(sentence)
  out_text = itos(Q.vocab.itos, out)
  print('input = [{0}]'.format(sentence))
  print('output = [{0}]'.format(out_text))
  return out_text

out = predict('우리 내일 같이 영화 볼래?')
out = predict('3박4일 놀러가고 싶다')
out = predict('가족끼리 여행간다.')